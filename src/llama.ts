// llama-server lifecycle: detect binary, spawn with the chosen GGUF, poll
// /health, and expose a stop() for graceful shutdown on `/models` swap or
// session end.

import { type ChildProcess, execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// Where we persist the PID of the currently-running llama-server so we can
// reap orphans on the next pi startup (if pi was SIGKILL'd or hard-crashed,
// our in-process exit hooks never fired).
const PID_FILE = join(homedir(), ".cache", "pi-llamacpp", "server.pid");

function writePidFile(pid: number): void {
	mkdirSync(dirname(PID_FILE), { recursive: true });
	writeFileSync(PID_FILE, String(pid), "utf8");
}

function clearPidFile(): void {
	try {
		rmSync(PID_FILE, { force: true });
	} catch {
		// swallow — clean shutdown shouldn't fail on this
	}
}

/**
 * Kill any llama-server left over from a previous pi run that crashed or was
 * SIGKILL'd. Idempotent; safe to call before every spawn. Only touches the
 * PID we wrote ourselves — other llama-server processes on the system are
 * left alone.
 */
export function reapOrphanServer(): void {
	if (!existsSync(PID_FILE)) return;
	let pid: number;
	try {
		pid = Number(readFileSync(PID_FILE, "utf8").trim());
	} catch {
		clearPidFile();
		return;
	}
	if (!Number.isFinite(pid) || pid <= 0) {
		clearPidFile();
		return;
	}
	try {
		// `kill 0` probes existence without signalling. Throws ESRCH if gone.
		process.kill(pid, 0);
		process.kill(pid, "SIGTERM");
		setTimeout(() => {
			try {
				process.kill(pid, 0);
				process.kill(pid, "SIGKILL");
			} catch {
				// already dead
			}
		}, 2000).unref();
	} catch {
		// process doesn't exist anymore; stale pid file
	}
	clearPidFile();
}

// Track every live llama-server child so we can reap them on ANY pi exit path
// (normal quit, uncaught exception, SIGINT/SIGTERM/SIGHUP). Our
// `session_shutdown` handler is the primary graceful path; this is belt +
// suspenders for the cases where it doesn't run — reload failures, hard
// crashes, external `kill` signals that don't propagate to children.
const liveChildren = new Set<ChildProcess>();
let exitHooksInstalled = false;

function installExitHooks(): void {
	if (exitHooksInstalled) return;
	exitHooksInstalled = true;

	// Synchronous last-resort kill. Runs on `process.exit()` and normal
	// shutdown; can only do sync work. `proc.kill()` IS sync (sends signal).
	process.on("exit", () => {
		for (const child of liveChildren) {
			if (child.exitCode === null) child.kill("SIGKILL");
		}
	});

	// For signal exits, Node's default handler would terminate without firing
	// "exit" for async cleanup. We install these to get one shot at a graceful
	// SIGTERM before the process dies. Re-raising the signal after cleanup
	// ensures Node reports the correct exit code.
	for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
		process.on(sig, () => {
			for (const child of liveChildren) {
				if (child.exitCode === null) child.kill("SIGTERM");
			}
			// Don't re-raise: if pi also has a handler for this signal (it
			// does for graceful /quit), we'd double-handle. Our SIGTERM to
			// the child is enough; pi will then call our session_shutdown.
		});
	}
}

export interface LlamaServerHandle {
	baseUrl: string;
	modelPath?: string;
	modelsDir?: string;
	contextSize: number;
	port: number;
	stop(): Promise<void>;
}

export class LlamaServerError extends Error {}

export function hasLlamaServer(): boolean {
	try {
		execFileSync("llama-server", ["--version"], { stdio: "ignore", timeout: 3000 });
		return true;
	} catch {
		return false;
	}
}

async function findFreePort(): Promise<number> {
	return new Promise((resolve, reject) => {
		const srv = createServer();
		srv.unref();
		srv.on("error", reject);
		srv.listen(0, () => {
			const addr = srv.address();
			if (addr && typeof addr === "object") {
				const port = addr.port;
				srv.close(() => resolve(port));
			} else {
				srv.close(() => reject(new Error("could not get port")));
			}
		});
	});
}

async function waitForHealth(baseUrl: string, timeoutMs: number, signal?: AbortSignal): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		if (signal?.aborted) throw new Error("aborted");
		try {
			const res = await fetch(`${baseUrl}/health`, { signal });
			if (res.ok) return;
			// 503 means "loading model" — keep polling.
		} catch {
			// connection refused while it starts; keep polling
		}
		await new Promise((r) => setTimeout(r, 500));
	}
	throw new LlamaServerError(`llama-server did not become ready within ${timeoutMs}ms`);
}

export interface SpawnOptions {
	// Exactly one of these must be set. `modelPath` => single-model mode (-m).
	// `modelsDir` => router-server mode (--models-dir): llama-server discovers
	// every GGUF under the dir and serves them all via /v1/models, loading on
	// demand. We use router mode so pi sees the catalog without us scanning.
	modelPath?: string;
	modelsDir?: string;
	contextSize: number;
	gpuLayers: number; // 999 = full offload; 0 = CPU only
	extraArgs?: string[];
	onStderr?: (chunk: string) => void;
}

export async function spawnLlamaServer(opts: SpawnOptions): Promise<LlamaServerHandle> {
	if (!hasLlamaServer()) {
		throw new LlamaServerError(
			"`llama-server` not found in PATH. Install llama.cpp: `brew install llama.cpp` (macOS) or https://github.com/ggml-org/llama.cpp",
		);
	}
	if (!opts.modelPath === !opts.modelsDir) {
		throw new LlamaServerError("spawnLlamaServer: exactly one of modelPath or modelsDir must be set");
	}

	const port = await findFreePort();
	const baseUrl = `http://127.0.0.1:${port}`;
	const args: string[] = [
		"-c",
		String(opts.contextSize),
		"--host",
		"127.0.0.1",
		"--port",
		String(port),
	];
	if (opts.modelPath) args.unshift("-m", opts.modelPath);
	else if (opts.modelsDir) args.unshift("--models-dir", opts.modelsDir);
	if (opts.gpuLayers > 0) args.push("-ngl", String(opts.gpuLayers));
	if (opts.extraArgs?.length) args.push(...opts.extraArgs);

	installExitHooks();
	const proc: ChildProcess = spawn("llama-server", args, { stdio: ["ignore", "pipe", "pipe"] });
	liveChildren.add(proc);
	if (proc.pid) writePidFile(proc.pid);
	proc.once("exit", () => {
		liveChildren.delete(proc);
		clearPidFile();
	});

	let lastStderr = "";
	proc.stderr?.on("data", (chunk: Buffer) => {
		const text = chunk.toString("utf8");
		lastStderr = text.split("\n").slice(-5).join("\n"); // keep tail for error messages
		opts.onStderr?.(text);
	});

	const exitPromise = new Promise<never>((_, reject) => {
		proc.once("exit", (code) => {
			reject(new LlamaServerError(`llama-server exited with code ${code}. Last stderr:\n${lastStderr}`));
		});
		proc.once("error", (err) => reject(new LlamaServerError(`failed to spawn llama-server: ${err.message}`)));
	});

	try {
		await Promise.race([waitForHealth(baseUrl, 120_000), exitPromise]);
	} catch (err) {
		if (!proc.killed) proc.kill("SIGTERM");
		throw err;
	}

	let stopped = false;
	const stop = async (): Promise<void> => {
		if (stopped || proc.exitCode !== null) return;
		stopped = true;
		proc.removeAllListeners("exit");
		proc.removeAllListeners("error");
		await new Promise<void>((resolve) => {
			proc.once("exit", () => resolve());
			proc.kill("SIGTERM");
			setTimeout(() => {
				if (proc.exitCode === null) proc.kill("SIGKILL");
			}, 3000).unref();
		});
	};

	return { baseUrl, modelPath: opts.modelPath, modelsDir: opts.modelsDir, contextSize: opts.contextSize, port, stop };
}
