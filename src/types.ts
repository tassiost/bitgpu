// Public types for bitgpu.

/** A byte range inside the data or aux file. */
export interface ManifestRef {
  src?: string
  dtype: string
  shape?: number[]
  off: number
  len: number
}
/** One tensor entry in a manifest. */
export interface ManifestTensor {
  kind?: string
  N?: number
  K?: number
  rows?: number
  cols?: number
  block?: number
  bits?: number
  lut?: string
  /** v2 manifests: 'q1_0' marks a tensor whose weight ref covers an interleaved GGUF
   *  Q1_0 region ([f16 scale][16 sign bytes] per 128-weight block); the loader demuxes
   *  it in-flight into the planar sign/scale buffers the kernels consume. */
  container?: string
  /** normalized at load: the raw interleaved region of a container tensor */
  q1_0?: ManifestRef
  weight?: ManifestRef
  scales?: ManifestRef
  zp?: ManifestRef
  // cos_cache / sin_cache are stored as bare refs:
  src?: string
  dtype?: string
  off?: number
  len?: number
}
/** The architecture contract of a manifest. */
export interface ManifestArch {
  model_type?: string
  layers: number
  hidden: number
  intermediate: number
  heads: number
  kv_heads: number
  head_dim: number
  rms_eps: number
  vocab: number
  eos: number
  act: string
  tie_word_embeddings?: boolean
  /** rope parameters for manifests without baked cos/sin caches (v2/GGUF) */
  rope?: { rope_theta: number; rope_type?: string; factor?: number; original_max_position_embeddings?: number }
  /** position cap for synthesized rope (GGUF context_length) */
  max_positions?: number
  /** Hybrid backbone (qwen3_5 / GGUF `qwen35`): a per-layer mix of gated-DeltaNet linear
   *  attention and gated full attention. Absent for the dense qwen3 models. When present, the
   *  full-attention layers use partial RoPE ({@link HybridArch.rotary_dim}) and an output gate. */
  hybrid?: HybridArch
  /** Vision tower config (Qwen3-VL). Only present on multimodal models (Bonsai-27B).
   *  When absent, the engine skips all vision initialization — zero cost for text-only models. */
  vision?: VisionConfig
}

/** Qwen3-VL vision tower configuration. Only present on multimodal models (Bonsai-27B).
 *  The vision tower is a separate ~0.63GB mmproj pack loaded only when an image arrives.
 *  See BITGPU_VISION_TOWER_PLAN.md for the full architecture. */
export interface VisionConfig {
  /** Number of transformer blocks in the ViT (27 for Bonsai-27B). */
  depth: number
  /** ViT hidden dimension (1152). */
  hidden_size: number
  /** MLP intermediate dimension (4304). */
  intermediate_size: number
  /** Number of attention heads (16). */
  num_heads: number
  /** Head dimension (72 = 1152 / 16). */
  head_dim: number
  /** Spatial patch size (16x16 pixels per patch). */
  patch_size: number
  /** Temporal patch size (2 frames per temporal patch). */
  temporal_patch_size: number
  /** Spatial merge size — merges 2x2 patches into 1 token (2). */
  spatial_merge_size: number
  /** Output dimension after the patch merger, matching the LLM hidden dim (3584 → projected to LLM hidden). */
  out_hidden_size: number
  /** Input channels (3 = RGB). */
  in_channels: number
  /** Number of learned position embeddings (2304 = 48x48 grid). */
  num_position_embeddings: number
  /** Token ID for the image placeholder token (248056). */
  image_token_id: number
  /** Token ID for <|vision_start|> (248053). */
  vision_start_token_id: number
  /** Token ID for <|vision_end|> (248054). */
  vision_end_token_id: number
  /** DeepStack: ViT layers whose features are injected into early LLM layers ([8, 16, 24]). */
  deepstack_visual_indexes?: number[]
  /** mrope_section for the vision RoPE: [temporal_dims, height_dims, width_dims]. */
  mrope_section?: [number, number, number]
}
/** The extra architecture contract for the hybrid qwen3_5 backbone. `heads`/`kv_heads`/`head_dim`
 *  on {@link ManifestArch} describe the FULL-attention layers; the linear layers are described here. */
export interface HybridArch {
  /** Per-layer token mixer, length = {@link ManifestArch.layers}: `'linear'` = gated DeltaNet,
   *  `'full'` = gated attention. Derived from the 3:1 `full_attention_interval` pattern. */
  layer_types: ('linear' | 'full')[]
  /** Number of key/query heads in the gated-DeltaNet layers (GGUF `ssm.group_count`). */
  linear_key_heads: number
  /** Number of value heads (>= key heads; query/key are repeat-interleaved to match). GGUF `ssm.time_step_rank`. */
  linear_value_heads: number
  /** Head dim of the linear-attention keys and values (they share it). GGUF `ssm.state_size`. */
  linear_head_dim: number
  /** Depthwise causal conv1d kernel width applied to the q/k/v stream (GGUF `ssm.conv_kernel`). */
  conv_kernel: number
  /** Rotary dims actually rotated in the full-attention layers (partial RoPE, GGUF `rope.dimension_count`);
   *  the remaining `head_dim - rotary_dim` dims pass through unrotated. */
  rotary_dim: number
}
/** The small model-description file bitgpu loads (see docs/FORMAT.md). Usually fetched as
 *  `manifest.json`; `bitgpu/gguf` builds one in memory straight from a GGUF header. */
export interface Manifest {
  version?: number
  data_file: string
  aux_file: string
  arch: ManifestArch
  luts: Record<string, ManifestRef>
  tensors: Record<string, ManifestTensor>
  /** Vision tower tensors (separate from language model tensors).
   *  Only present when `arch.vision` is set. Loaded from the mmproj pack. */
  visionTensors?: Record<string, ManifestTensor>
}

/** Progress event emitted while a model loads. */
export interface LoadProgress {
  phase: 'manifest' | 'weights' | 'pipelines'
  /** Bytes fetched so far (weights phase only). */
  loaded?: number
  /** Total bytes to fetch (weights phase only). */
  total?: number
}

/** Options for {@link createEngine}. Pass a string to use defaults with just a model URL.
 *  Provide either `modelUrl` (a directory holding manifest.json + the data/aux files) or the explicit
 *  `manifestUrl`/`dataUrl`/`auxUrl` (which lets the large data file come from one host, e.g. the HF Hub,
 *  and the small manifest/aux from another). The explicit URLs win over the `modelUrl`-relative ones. */
export interface EngineOptions {
  /** Base URL of the model directory containing `manifest.json` and its data/aux files. */
  modelUrl?: string
  /** Explicit URL for manifest.json (overrides `${modelUrl}/manifest.json`). */
  manifestUrl?: string
  /** Explicit URL for the weights data file (overrides `${modelUrl}/<data_file>`). */
  dataUrl?: string
  /** Explicit URL for the aux file (overrides `${modelUrl}/<aux_file>`). */
  auxUrl?: string
  /** An in-memory manifest (skips the manifest fetch). `bitgpu/gguf`'s `fromGguf` returns one
   *  parsed straight from a GGUF header. Requires `dataUrl` (or `modelUrl`) for the weights. */
  manifest?: Manifest
  /** In-memory aux bytes (skips the aux fetch). `fromGguf` computes them (the LUTs are derived,
   *  not stored, for GGUF models). */
  aux?: ArrayBuffer | Uint8Array
  /** Fetch JSON (manifest). Override to add caching. Default: `fetch(url).json()`. */
  fetchJson?: (url: string) => Promise<unknown>
  /** Fetch binary (data/aux). Override to add caching (e.g. OPFS) for the ~290MB data file. Default: `fetch(url).arrayBuffer()`. */
  fetchArrayBuffer?: (url: string) => Promise<ArrayBuffer>
  /** Stream the big weights DATA file: chunks flow straight into GPU buffers, so the whole
   *  ~290MB file never sits in memory at once (critical on phones, where the buffered peak gets
   *  the tab killed). Used for the data file only (manifest/aux stay on fetchJson/fetchArrayBuffer);
   *  when set it takes precedence over fetchArrayBuffer for that file. Point it at a cache stream,
   *  e.g. an OPFS file's `stream()`. Default: the data file streams from `fetch(url).body`
   *  (or through `fetchArrayBuffer` when that override is provided, for back compatibility). */
  fetchStream?: (url: string) => Promise<ReadableStream<Uint8Array>>
  /** GPU power preference. Default `'high-performance'` (picks the discrete GPU on multi-GPU machines).
   *  (Spelled out rather than `GPUPowerPreference` so the published d.ts resolves without @webgpu/types.) */
  powerPreference?: 'low-power' | 'high-performance'
  /** Force the no-subgroup reduction path (for testing the fallback). Default `false`. */
  forceNoSubgroups?: boolean
  /** Workgroup size for the no-subgroup reduction kernels. Default `64`. */
  noSubgroupWorkgroupSize?: number
  /** Vision tower mmproj URL — the separate ~0.63GB pack for the Qwen3-VL vision encoder.
   *  Only fetched when `manifest.arch.vision` is present AND an image is passed to the engine.
   *  For text-only generation, it's never fetched. Default: derived from `modelUrl` or `dataUrl`
   *  by replacing the filename with `Bonsai-27B-mmproj-Q8_0.gguf`. */
  visionMmprojUrl?: string
  /** Decode steps chained per CPU sync (deferred readback). Higher hides latency; default `4`. */
  syncSteps?: number
  /** Prefill GEMM tiling: `'auto'` tiles once a prompt fills the 64-row tiles, `'always'`/`'never'` force it. Default `'auto'`. */
  prefillTiling?: 'auto' | 'always' | 'never'
  /** Fuse RMSNorm into the decode matmul kernels (saves 2 dispatches/layer). Each matmul
   *  workgroup redundantly computes the sum-of-squares, but eliminates the separate
   *  rmsnorm_sg dispatch. Net win when kernel launch overhead > redundant compute.
   *  Default `false` (opt-in for benchmarking). Decode-only; prefill path unchanged. */
  fusedDecode?: boolean
  /** Run a dummy decode step during load to force GPU driver JIT compilation of all
   *  pipeline state objects. The first real decode token skips the driver's lazy native
   *  shader compilation, reducing TTFT. Mirrors the diffusion pipeline's "pre-run a tiny
   *  dummy textToImage to force shader JIT" pattern. Default `false` (opt-in).
   *  Cost: ~50-100ms during load (one decode step + readback + cache reset). */
  warmShaders?: boolean
  /** Merge the 3 post-QKV-matmul dispatches (Q norm+RoPE, K norm+RoPE+cache write,
   *  V cache write) into a single dispatch. Saves 2 dispatches per layer (56 for 28
   *  layers). Consumer merge — no redundant compute. kv8 cache + subgroup path only.
   *  Default `false` (opt-in for benchmarking). Decode-only; prefill path unchanged. */
  fusedQKV?: boolean
  /** Max KV-cache length (prompt + generated positions). Caps VRAM (~`maxSeqLen` x 224 KB at f32,
   *  half that with `kvCache: 'f16'`, ~a quarter with `'q8'`). Default `2048`. Capped by the
   *  model's RoPE range: the baked cache length for ONNX-derived manifests, `max_positions`
   *  (the GGUF context length) for GGUF-derived ones - exceeding it fails loudly at load. */
  maxSeqLen?: number
  /** KV-cache STORAGE precision. `'f16'` halves KV memory (all arithmetic stays f32; each cached
   *  K/V value is rounded once at cache-write); it falls back to `'f32'` silently when the
   *  adapter lacks `shader-f16`. `'q8'` QUARTERS KV memory vs f32 (packed 8-bit values with one
   *  f32 scale per 32-element block, llama.cpp q8_0-style, ~1.125 bytes/value) and needs no
   *  adapter feature, so it works even where f16 does not (and at long context it decodes FASTER
   *  than f32 - attention there is memory-bound and reads a quarter of the bytes). Outputs under
   *  f16/q8 are no longer
   *  guaranteed bit-identical to f32 mode - q8 is the first knowingly-lossy tier, measured per
   *  model by the GPU gate - though WITHIN a mode decoding stays exact and deterministic (same
   *  seed -> same tokens; cache reuse == full prefill). See `capabilities.kvCache` for what's
   *  active. Default `'f32'`. */
  kvCache?: 'f32' | 'f16' | 'q8'
  /** Activation-compute precision for the decode matmuls (the 1-bit GEMVs that dominate decode).
   *  `'f16'` reads activations as f16 and runs the per-block dot in f16 (2x ALU rate on
   *  Apple/AMD/recent NVIDIA) with f32 accumulation; the residual stream and weights stay f32.
   *  Like `kvCache: 'f16'` it needs the `shader-f16` adapter feature and the subgroup path, and
   *  SILENTLY FALLS BACK to f32 without them, so it is safe to request unconditionally. Outputs
   *  are not bit-identical to f32 (measured near-lossless by the GPU gate, q8-KV tier) but WITHIN
   *  the mode decoding is exact and deterministic. Applies to decode only (prefill stays f32).
   *  See `capabilities.activation` for what is active. Default `'f32'`. */
  activation?: 'f32' | 'f16'
  /** What happens when a conversation outgrows `maxSeqLen`. `'error'` (default): generate
   *  throws, exactly as before. `'sinks'`: StreamingLLM-style rolling window - the first
   *  `sinkTokens` positions (attention sinks) plus the most recent window are kept and the
   *  middle is evicted in batches, so generation and multi-turn chat continue indefinitely in
   *  fixed memory. Keys are cached UNROPED and rotated at read by cache-relative position, so
   *  eviction never rewrites or (under q8) requantizes cache bytes. Sink mode is its own
   *  measured mode like f16/q8: within-mode decoding stays exact and deterministic, and before
   *  the first eviction f32+sinks matches default f32 (gated); after eviction the model
   *  genuinely forgets evicted middle context - that is the trade. Prompts longer than the
   *  window still throw (trim prompt-side, e.g. chat onOverflow). */
  overflow?: 'error' | 'sinks'
  /** Number of initial attention-sink positions kept forever under `overflow: 'sinks'`.
   *  Default `4` (the StreamingLLM setting). */
  sinkTokens?: number
  /** Use GPU-side dequantization for Q1_0 weights instead of CPU-side per-byte
   *  transforms. Uploads raw GGUF bytes via writeBuffer (no CPU processing),
   *  then dispatches a compute shader to expand into the codes/signs + scales
   *  format the matmul kernels expect. ~2x faster weight upload. Default `false`. */
  gpuDequant?: boolean
  /** Batch all raw weight uploads into a single large writeBuffer call, then
   *  distribute to individual tensor buffers via GPU-side copyBufferToBuffer.
   *  Eliminates per-chunk writeBuffer IPC/shmem overhead (~3700 calls → 1).
   *  Requires gpuDequant. ~3-5x faster on top of gpuDequant. Default `false`. */
  batchUpload?: boolean
  /** Create raw weight buffers with mappedAtCreation and write directly into
   *  the mapped range instead of calling writeBuffer per chunk. On Apple Silicon
   *  UMA this can skip the staging buffer copy entirely. Requires gpuDequant.
   *  Default `false`. */
  mappedUpload?: boolean
  /** Benchmark vision tower: split the 27 layers into separate command buffers
   *  with CPU wall-clock timing per phase. Prints per-phase breakdown to console.
   *  Default `false` (opt-in for benchmarking). */
  benchVision?: boolean
  /** Called as the model loads. */
  onProgress?: (progress: LoadProgress) => void
  /** Called if the GPU device is lost after creation (driver reset, OS reclaim, tab backgrounding
   *  on some platforms). The engine is unusable afterward; create a new one to recover. Not called
   *  for the intentional loss caused by {@link Engine.dispose}. */
  onDeviceLost?: (info: DeviceLostInfo) => void
}

/** Why the GPU device went away. Mirrors GPUDeviceLostInfo without requiring @webgpu/types. */
export interface DeviceLostInfo {
  /** 'destroyed' when {@link Engine.dispose} caused it, otherwise the platform reason ('unknown', ...). */
  reason: string
  message: string
}

/** Options for a single {@link Engine.generate} call. Set `temperature` to a value other than 0 or 1
 *  to sample; otherwise decoding is greedy (argmax). Sampling matches transformers.js v4.2.0 exactly
 *  (repetition_penalty, no_repeat_ngram, temperature, top_k, multinomial via a Mersenne-Twister RNG). */
export interface GenerateOptions {
  /** Maximum number of new tokens to generate. Default `256`. `0` prefills the prompt into the
   *  KV cache and emits nothing (like {@link Engine.prefill}, but composable with `reuseCache`). */
  maxTokens?: number
  /** Reuse the KV cache from the previous turn: treat the passed token ids as the DELTA to append to
   *  the cached conversation (not a full prompt), prefilling only those tokens. Requires a prior
   *  generate on this engine (else it falls back to a full prefill). Default `false` (reset + full prefill). */
  reuseCache?: boolean
  /** Token ids that end generation when produced (e.g. EOS). The stop token is not emitted. */
  stopTokens?: number[]
  /** Called with each generated token id as it is produced (per-token when sampling). */
  onToken?: (tokenId: number) => void
  /** Aborts generation when signaled (checked at each step). */
  signal?: AbortSignal
  /** Softmax temperature. A value other than 0 or 1 enables sampling; 0 or 1 (or unset) is greedy. */
  temperature?: number
  /** Top-k sampling cutoff (candidate count). Default `20` when sampling; clamped to `[1, vocab]`
   *  (the GPU reduces the logits to this many candidates, so `0` cannot mean "disabled"). */
  topK?: number
  /** Top-p (nucleus) cutoff, applied over the top-K candidates when sampling: keeps the shortest
   *  leading run whose cumulative probability reaches `topP`. Default `1` (off). Only affects
   *  sampling (greedy takes the penalized argmax regardless). */
  topP?: number
  /** Min-p cutoff, applied over the top-K candidates when sampling: keeps tokens whose probability
   *  is at least `minP` * (the top token's probability), so the pool tightens when the model is
   *  confident and widens when it is not. Default `0` (off). Robust for low-precision models. Only
   *  affects sampling. */
  minP?: number
  /** Repetition penalty over the deduped prompt+generated id set (`logit<0 ? *p : /p`). Default `1`
   *  (off). Applied under greedy decoding too (the penalized argmax), matching transformers.js. */
  repetitionPenalty?: number
  /** Presence penalty: subtracts this flat amount from the logit of every token seen so far (the
   *  additive anti-repetition knob the Qwen3.5 family recommends, applied after `repetitionPenalty`).
   *  Default `0` (off). Applied on the full vocab before top-k, under greedy decoding too. */
  presencePenalty?: number
  /** DRY ("don't repeat yourself") repetition penalty strength. Default `0` (off). When a candidate
   *  token would EXTEND a repeat - the last L history tokens followed by it reproduce an earlier
   *  stretch - and L >= `dryAllowedLength`, its logit drops by
   *  `dryMultiplier * dryBase^(L - dryAllowedLength)`. Targets loops that per-token penalties miss
   *  while leaving legitimately repeated single tokens alone. Applied on the CPU over the top-K
   *  candidates (like `topP`/`minP`), under greedy decoding too; reported logprobs are pre-DRY.
   *  Not supported together with `promptLookup` (with `'auto'`, lookup simply stays off). */
  dryMultiplier?: number
  /** DRY exponential base (default `1.75`). */
  dryBase?: number
  /** Repeat length that DRY tolerates before penalizing (default `2`). */
  dryAllowedLength?: number
  /** DRY only searches the last `dryRange` history tokens (default `0` = the whole history). */
  dryRange?: number
  /** Token ids that act as DRY sequence breakers: matching never crosses them and they are never
   *  penalized themselves (structural tokens - newlines, quotes, list markers - legitimately
   *  repeat). Default none. `bitgpu/chat` fills this with the tokenizer's newline/punctuation
   *  tokens when DRY is enabled. */
  dryBreakers?: number[]
  /** Top-n-sigma sampling cutoff (arXiv 2411.07641): keep candidates whose logit is within
   *  `topNSigma` standard deviations of the max, with sigma computed on the GPU over the FULL
   *  penalized logit vector (not a top-K estimate). Robust for low-precision models: the
   *  threshold adapts to how peaked the distribution actually is. Typical value 1.0-1.5; default
   *  `0` (off, bit-identical draw). Only affects sampling; not combinable with `promptLookup`
   *  (with `'auto'`, lookup simply stays off). */
  topNSigma?: number
  /** Block any n-gram of this size from repeating. Default `0` (off). Applied under greedy decoding
   *  too, matching transformers.js. */
  noRepeatNgramSize?: number
  /** Seed for the sampler RNG. Omit to seed from entropy (non-deterministic, like production). */
  seed?: number
  /** EXPERIMENTAL. Per-step constrained-decoding hook: receives the top-K candidate token ids
   *  (rank order, penalized logits alongside) and returns the PERMITTED subset (any order).
   *  Greedy picks the best permitted candidate; sampling renormalizes the draw over them. When
   *  it permits none, the engine walks the full vocabulary in logit order (rare; the step costs
   *  a full logits readback) and throws if no token at all is permitted. Must be deterministic
   *  and synchronous; the engine calls it once per emitted token, in order. Incompatible with
   *  `promptLookup` (speculation is disabled while a filter is set). Used by bitgpu/chat's
   *  `format: 'json'`. */
  candidateFilter?: (candidateIds: Uint32Array, candidateLogits: Float32Array) => number[]
  /** EXPERIMENTAL. Prompt-lookup speculative decoding: draft the continuation from an n-gram
   *  match in the sequence so far and verify every draft in ONE batched forward. Output is
   *  identical to normal decoding, greedy AND sampled (each emitted token still comes from its
   *  true penalized distribution, and the RNG advances one draw per emitted token, in order).
   *  `'auto'` speculates for a short probation window, then keeps PLD only when the measured
   *  acceptance actually beats plain decoding for this content - and falls back to the plain
   *  path otherwise (output still identical; `speculation.bailed` reports the decision). Use
   *  `'auto'` unless you know the content repeats (quoting, lists, code), where `true` skips
   *  the probation. INCOMPATIBLE with `noRepeatNgramSize`: a prompt-lookup draft is the
   *  continuation of a repeated n-gram, which is exactly what the ngram ban forbids, so
   *  acceptance is ~zero when both are on (measured 0/320). No draft model, no extra VRAM.
   *  `true`/`'auto'` = `{ ngramSize: 3, maxDraft: 8 }`. Default `false`. */
  promptLookup?: boolean | 'auto' | { ngramSize?: number; maxDraft?: number }
  /** Per-token TRUE logprobs (log-softmax over the full vocabulary, after penalties): set to N
   *  (1..32) to receive the emitted token's logprob plus the top-N alternatives each step in
   *  {@link GenerateResult.logprobs}. Exact, not a top-K approximation - the normalizer is a GPU
   *  log-sum-exp over all logits (one extra f32 readback per step). Routes greedy turns through
   *  the per-step sampler path and disables `promptLookup` (a verified draft step has no
   *  per-token candidate readback); sampled turns pay nothing extra. Use it to expose model
   *  confidence: a low top-1 logprob or a flat top-N is the model guessing. Default off. */
  logprobs?: number
}

/** One emitted token's logprob record (see {@link GenerateOptions.logprobs}). */
export interface TokenLogprobs {
  /** The emitted token's logprob (log-softmax over the full vocab, post-penalty). Under a
   *  candidateFilter this is the logprob of the token actually chosen, which the filter may
   *  have forced far below the top alternatives. */
  logprob: number
  /** The top-N (id, logprob) pairs in descending order, independent of what was emitted. */
  top: { id: number; logprob: number }[]
}

/** Result of a {@link Engine.generate} call. */
export interface GenerateResult {
  /** Generated token ids (excludes the prompt). */
  tokens: number[]
  /** Time to first token (prefill of the prompt), in milliseconds. */
  prefillMs: number
  /** Decode time for the remaining tokens, in milliseconds. */
  decodeMs: number
  /** Decode throughput (tokens / second), excluding prefill. */
  tokensPerSecond: number
  /** Per-token decode timing breakdown, in milliseconds. In sampled mode the GPU wait and the
   *  readback map share one sync, so the wait is attributed to `gpuMs` and `readbackMs` covers
   *  only the post-map CPU work (near zero). */
  timing: { recordMs: number; gpuMs: number; readbackMs: number }
  /** Present when prompt-lookup decoding ran: verify steps taken, tokens drafted, drafts
   *  accepted. With `promptLookup: 'auto'`, `bailed` reports whether the probation decided to
   *  stop speculating for the rest of the turn. */
  speculation?: { steps: number; drafted: number; accepted: number; bailed?: boolean }
  /** Present when {@link GenerateOptions.logprobs} was set: one record per emitted token,
   *  aligned with `tokens`. */
  logprobs?: TokenLogprobs[]
}

/** Diagnostic result of {@link Engine.forward}: hidden states + logits for a single forward pass. */
export interface ForwardResult {
  embed: Float32Array
  layer0: Float32Array
  finalnorm: Float32Array
  logits: Float32Array
  vocab: number
  sequenceLength: number
}

/** Image input for the vision tower. The image is preprocessed on CPU
 *  (resized to a multiple of patch_size, converted to RGB float tensor)
 *  before being passed to the engine. */
export interface ImageInput {
  /** RGB pixel data, row-major, [C, H, W] layout (C=3).
   *  Values normalized to [0, 1] (pixel / 255). */
  rgb: Float32Array
  /** Image width in pixels (must be a multiple of patch_size * spatial_merge_size). */
  width: number
  /** Image height in pixels (must be a multiple of patch_size * spatial_merge_size). */
  height: number
  /** Number of frames (1 for single image, >1 for video). Default: 1. */
  frames?: number
}

/** Result of {@link Engine.visionForward}: image embeddings ready to insert
 *  into the token sequence, plus DeepStack features for early LLM layer injection. */
export interface VisionForwardResult {
  /** Final image embeddings [num_merged_patches, out_hidden_size].
   *  These replace the image_token_id positions in the token embedding sequence. */
  imageEmbeds: Float32Array
  /** Number of merged patches (= imageEmbeds.length / out_hidden_size). */
  numPatches: number
  /** DeepStack features from ViT layers [8, 16, 24].
   *  Each entry is [num_merged_patches, out_hidden_size].
   *  Injected into LLM layers 0, 1, 2 respectively. Empty if DeepStack is not used. */
  deepstackFeatures: Float32Array[]
  /** Time taken in milliseconds. */
  elapsedMs: number
}

/** What the engine detected about the host GPU and which code path it selected. */
export interface EngineCapabilities {
  /** Whether the fast subgroup path is in use (false = the workgroup-reduction fallback). */
  useSubgroups: boolean
  /** Subgroup width when the subgroup path is active. */
  subgroupSize: number
  /** Active KV-cache storage precision ('f16' only when requested AND the adapter has shader-f16;
   *  'q8' whenever requested - it needs no adapter feature). */
  kvCache: 'f32' | 'f16' | 'q8'
  /** Active activation-compute precision ('f16' only when requested AND the adapter has shader-f16
   *  AND the subgroup path is in use; else 'f32'). */
  activation: 'f32' | 'f16'
  /** Active overflow policy ('sinks' = rolling window with attention sinks). */
  overflow: 'error' | 'sinks'
  /** The engine's KV window in positions (the resolved maxSeqLen option). Under
   *  `overflow: 'error'` prompt + generated tokens must fit inside it; under `'sinks'` it is
   *  the rolling window size. */
  maxSeqLen: number
  /** Adapter identification, when the browser exposes it. */
  adapter: { vendor?: string; architecture?: string; device?: string; description?: string }
  /** Relevant adapter limits the engine codes against. */
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupStorageSize: number }
  /** Whether the device has the `timestamp-query` feature (true GPU-side kernel timing is
   *  available to the dev profiler; see the profileDecode diagnostics). */
  timestampQuery: boolean
}

/** A saved KV cache + conversation token history, from {@link Engine.saveCache}. A plain object
 *  holding one `ArrayBuffer`, so it is structured-cloneable: store it in IndexedDB / OPFS or send
 *  it over `postMessage` as-is (it is NOT `JSON.stringify`-able - the buffer would be lost).
 *  Restore requires the SAME model architecture and the SAME `kvCache` mode; snapshots never
 *  convert across modes. Treat the fields as opaque. */
export interface KvSnapshot {
  /** Snapshot format version: `1` = default mode, `2` = saved under `overflow: 'sinks'`
   *  (the cache holds unroped keys and a rolled window; restore requires sink mode). */
  version: 1 | 2
  /** KV storage mode the snapshot was taken under. */
  kvCache: 'f32' | 'f16' | 'q8'
  /** Version 2 only: rolling-window state (sink count and filled cache slots). */
  roll?: { sinkTokens: number; cacheLen: number }
  /** Architecture stamp; restore rejects a snapshot from a different model shape. */
  model: { layers: number; kvHeads: number; headDim: number; hidden: number; vocab: number }
  /** The full conversation token sequence at save time. */
  ids: number[]
  /** DELTA snapshot: `data` holds only cache positions `[base, len)` - the leading `base` positions
   *  (e.g. a shared prewarmed system prompt) are NOT stored and must already be present in the
   *  restore target (restore into an engine freshly prewarmed with the same prefix). Absent/0 = a
   *  full snapshot (all positions in `data`). Restore validates the target is exactly at the base
   *  boundary with a matching token prefix. */
  base?: number
  /** Packed cached K/V bytes (and q8 block scales) for the stored positions, per KV-bearing layer;
   *  for the qwen3_5 hybrid the DeltaNet recurrent + conv state of each linear layer is appended. */
  data: ArrayBuffer
}

/** A loaded model ready to generate. Create one with {@link createEngine}. */
export interface Engine {
  /** Generate tokens from a prompt given as token ids. */
  generate(promptTokenIds: number[], options?: GenerateOptions): Promise<GenerateResult>
  /** Prefill a prompt PREFIX into the KV cache without decoding, so a later
   *  `generate(delta, { reuseCache: true })` continues from it. Use to warm a static system prompt
   *  at load time so the first real turn is a cheap cache-append, not a full prefill. Resets any
   *  existing cache (the prefix becomes the whole history). Returns the prefill wall-time. */
  prefill(promptTokenIds: number[]): Promise<{ prefillMs: number }>
  /** Run a single forward pass and return hidden states + logits (diagnostic / correctness checks). */
  forward(tokenIds: number[]): Promise<ForwardResult>
  /** Run the vision tower on one or more images and return image embeddings + DeepStack features.
   *  Throws if the model has no vision tower (`arch.vision` is undefined).
   *  The mmproj pack is loaded lazily on first call (if not already loaded). */
  visionForward?(images: ImageInput[]): Promise<VisionForwardResult>
  /** Multimodal generate: run the vision tower on `images`, inject the resulting embeddings
   *  at `imagePositions` in the token sequence (the `image_token_id` placeholder positions),
   *  then decode as normal. The vision tower's projection_dim must match the LLM hidden_size
   *  (it does for Bonsai-27B: both are 5120). Throws if the model has no vision tower.
   *  `imagePositions` must be sorted ascending and each < promptTokenIds.length. */
  generateWithImages?(promptTokenIds: number[], imagePositions: number[], images: ImageInput[], options?: GenerateOptions): Promise<GenerateResult>
  /** Whether the model has a vision tower loaded and ready. `false` for text-only models
   *  (1.7B/4B/8B) and for the 27B before the mmproj pack is fetched. */
  readonly vision?: boolean
  /** Clear the cross-turn KV cache and token history (start a fresh conversation). */
  resetCache(): void
  /** Snapshot the current conversation - KV cache contents + token history - as a
   *  structured-cloneable {@link KvSnapshot} (GPU -> CPU readback). Restoring it, into this
   *  engine or a fresh one on the same model and `kvCache` mode, is bit-identical to having kept
   *  the conversation alive, so conversations survive engine disposal and page reloads (persist
   *  the snapshot in IndexedDB / OPFS). Size ~ tokens x layers x kvHeads x headDim x 2 x
   *  bytes-per-value (~224 KB/token at f32 on Bonsai-1.7B, ~63 KB/token at q8). Returns `null`
   *  when the cache is empty. Serialized with generate/prefill/forward like every engine op.
   *
   *  `from` makes a DELTA snapshot that excludes the first `from` cache positions (a shared
   *  prewarmed prefix), so per-conversation snapshots drop the redundant system-prompt KV (tens of
   *  MB at chat scale). Restore it into an engine freshly prewarmed with the SAME prefix - see the
   *  `base` field on {@link KvSnapshot}. `from` counts cache positions: for a prewarm of P tokens
   *  pass `from = P - 1` (a fresh prewarm caches P-1 positions; the last prewarm token's K/V is part
   *  of the delta). `bitgpu/chat`'s `save({ delta: true })` computes this for you. */
  saveCache(opts?: { from?: number }): Promise<KvSnapshot | null>
  /** Replace the current conversation with a {@link KvSnapshot} (CPU -> GPU upload). The next
   *  `generate(delta, { reuseCache: true })` continues the saved conversation exactly. Throws if
   *  the snapshot's model architecture or `kvCache` mode does not match this engine, or if it
   *  exceeds this engine's `maxSeqLen`. */
  restoreCache(snapshot: KvSnapshot): Promise<void>
  /** Detected GPU capabilities and selected code path. */
  readonly capabilities: EngineCapabilities
  /** Resolves when the GPU device is lost (including via {@link dispose}, with reason 'destroyed').
   *  After an unexpected loss the engine is dead; create a new one to recover. */
  readonly lost: Promise<DeviceLostInfo>
  /** Release GPU resources. The engine is unusable afterward. */
  dispose(): void
}
