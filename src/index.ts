// pi-llamacpp — registers a `llama-cpp` provider backed by a long-lived
// llama-server in router mode. The server is spawned once at extension init,
// pointed at the llama.cpp on-disk cache via `--models-dir`. We then read the
// catalog from `/v1/models` instead of scanning the filesystem ourselves.
//
// Models dir (in priority order):
//   1. $PI_LLAMACPP_MODELS_DIR if set
//   2. llama.cpp's own cache (where `llama-server -hf …` drops files):
//        macOS:   ~/Library/Caches/llama.cpp
//        Linux:   $XDG_CACHE_HOME/llama.cpp  (fallback ~/.cache/llama.cpp)
//        Windows: %LOCALAPPDATA%/llama.cpp
//
// Trade-offs vs. the previous per-pick spawn:
//   - No per-model native-context probe — `-c` becomes a server-wide cap.
//     Set $PI_LLAMACPP_CONTEXT to tune (default 32768).
//   - Sampling defaults (--temp, --top-p, …) apply to every served model.
//     Per-request overrides via the OpenAI API still work.
//   - Single server process → single restart cost; model switches happen
//     inside llama-server via `--models-autoload` (no spawn/stop dance).

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import type { ExtensionAPI, ProviderModelConfig } from "@mariozechner/pi-coding-agent";

import { type LlamaServerHandle, reapOrphanServer, spawnLlamaServer } from "./llama.js";

const PROVIDER = "llama-cpp";

// Server-wide context cap. Models with a larger native context are clamped to
// this; smaller models are unaffected. Must exceed pi's compaction reserve.
const CONTEXT_CAP = Number(process.env.PI_LLAMACPP_CONTEXT) || 32768;

function defaultModelsDir(): string | null {
	if (process.platform === "darwin") return join(homedir(), "Library", "Caches", "llama.cpp");
	if (process.platform === "win32") {
		return process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, "llama.cpp") : null;
	}
	const xdg = process.env.XDG_CACHE_HOME || join(homedir(), ".cache");
	return join(xdg, "llama.cpp");
}

interface OpenAIModelList {
	object: "list";
	data: Array<{ id: string; object: string; created?: number; owned_by?: string }>;
}

export default async function pilocalExtension(pi: ExtensionAPI): Promise<void> {
	reapOrphanServer();

	const modelsDir = process.env.PI_LLAMACPP_MODELS_DIR || defaultModelsDir();
	if (!modelsDir || !existsSync(modelsDir)) {
		console.warn(
			`[llama-cpp] models dir not found: ${modelsDir ?? "(unset)"}. ` +
				"Seed it with `llama-server -hf <repo>:<quant>` once, or set $PI_LLAMACPP_MODELS_DIR.",
		);
		return;
	}

	let server: LlamaServerHandle | null = null;
	try {
		server = await spawnLlamaServer({
			modelsDir,
			contextSize: CONTEXT_CAP,
			gpuLayers: 999,
			// Sampling flags (--temp, --top-p, …) deliberately omitted: in router
			// mode the children do inference, not this process, so flags here are
			// no-ops. Pi sends sampling params per-request; per-model defaults
			// live in the preset INI alongside each GGUF's manifest.
		});
	} catch (err) {
		console.error(`[llama-cpp] failed to start router server: ${describeError(err)}`);
		return;
	}

	let modelIds: string[];
	try {
		modelIds = await fetchModelIds(server.baseUrl);
	} catch (err) {
		console.error(`[llama-cpp] /v1/models failed: ${describeError(err)}`);
		await server.stop().catch(() => {});
		return;
	}

	if (modelIds.length === 0) {
		console.warn(`[llama-cpp] router server returned no models. Is ${modelsDir} empty?`);
	}

	pi.registerProvider(PROVIDER, {
		baseUrl: `${server.baseUrl}/v1`,
		apiKey: "none",
		api: "openai-completions",
		models: modelIds.map(toModelDef),
	});

	pi.on("session_shutdown", async () => {
		if (!server) return;
		const s = server;
		server = null;
		await s.stop().catch(() => {});
	});
}

async function fetchModelIds(baseUrl: string): Promise<string[]> {
	const res = await fetch(`${baseUrl}/v1/models`);
	if (!res.ok) throw new Error(`/v1/models returned ${res.status}`);
	const body = (await res.json()) as OpenAIModelList;
	return body.data.map((m) => m.id);
}

function toModelDef(id: string): ProviderModelConfig {
	return {
		id,
		name: id,
		reasoning: false,
		input: ["text"] as ("text" | "image")[],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: CONTEXT_CAP,
		maxTokens: CONTEXT_CAP,
		compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" as const },
	};
}

function describeError(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}
