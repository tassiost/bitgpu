// Vision tower forward pass for Bonsai-27B (Qwen3-VL).
//
// This module is loaded ONLY when the model has a vision tower (arch.vision).
// It implements the full vision pipeline:
//   1. Patch embedding (Conv3d → linear projection, 1536→1152)
//   2. Add interpolated position embeddings (bilinear from 48×48 learned grid)
//   3. 27× VisionBlock (LayerNorm → bidirectional attention → residual →
//      LayerNorm → GELU MLP → residual)
//   4. Post layernorm
//   5. Patch merger (pixel shuffle 2x2 → 4608 → linear(4608→4608) → GELU → linear(4608→5120))
//
// Weight formats (from the real mmproj GGUF):
//   - Q8_0: attn_qkv, attn_out, ffn_up, mm.0, mm.2 (8-bit quant: f16 scale + 32 int8 per block)
//   - F16:  ffn_down (half precision)
//   - F32:  biases, ln1/ln2 weight+bias, patch_embd, position_embd, post_ln
//
// NO DeepStack — is_deepstack_layers is all false in the mmproj metadata.
// NO HQQ 4-bit — the plan was wrong; weights are standard Q8_0 + F16.
//
// See BITGPU_VISION_TOWER_PLAN.md for the full architecture.

import type {
  VisionConfig, ImageInput, VisionForwardResult,
} from './types'

// ─── Q8_0 dequantization (CPU-side, for weight loading) ──────────────

/** Q8_0 block: 2-byte f16 scale + 32 int8 values = 34 bytes per 32 elements. */
const Q8_0_BLOCK_SIZE = 34

/** Dequantize a Q8_0 tensor to Float32Array.
 *  Q8_0 layout: [N, K/32 blocks], each block = f16 scale + 32 int8 values.
 *  Dequant: value = f32(int8_value) * f32(scale) */
export function dequantQ8_0(data: Uint8Array, N: number, K: number): Float32Array {
  const out = new Float32Array(N * K)
  const blocksPerRow = K / 32
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)

  for (let row = 0; row < N; row++) {
    for (let blk = 0; blk < blocksPerRow; blk++) {
      const blockOff = (row * blocksPerRow + blk) * Q8_0_BLOCK_SIZE
      // f16 scale (little-endian)
      const scaleBits = dv.getUint16(blockOff, true)
      const scale = f16ToF32(scaleBits)
      // 32 int8 values
      for (let i = 0; i < 32; i++) {
        const val = dv.getInt8(blockOff + 2 + i)
        out[row * K + blk * 32 + i] = val * scale
      }
    }
  }
  return out
}

/** Repack Q8_0 data into GPU-friendly format for in-shader dequantization.
 *  Returns two arrays:
 *  - packed: [N, K/4] u32 words, each containing 4 int8 values (little-endian)
 *  - scales: [N, K/32] f32 values, one scale per 32-element block
 *
 *  The shader reads packed words and scales, dequantizes inline:
 *    value = f32(int8_value) * scale
 *
 *  Weight bandwidth: K bytes/row (packed) + K/8 bytes/row (scales) = 1.125K
 *  vs 4K for f32 — 3.56x reduction. */
export function repackQ8_0(
  data: Uint8Array, N: number, K: number,
): { packed: Uint32Array, scales: Float32Array } {
  const blocksPerRow = K / 32
  const packed = new Uint32Array(N * (K / 4))   // 8 u32 words per block
  const scales = new Float32Array(N * blocksPerRow)
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)

  for (let row = 0; row < N; row++) {
    for (let blk = 0; blk < blocksPerRow; blk++) {
      const blockOff = (row * blocksPerRow + blk) * Q8_0_BLOCK_SIZE
      // f16 scale → f32
      scales[row * blocksPerRow + blk] = f16ToF32(dv.getUint16(blockOff, true))
      // 32 int8 values → 8 u32 words (reinterpret bytes as little-endian u32)
      for (let w = 0; w < 8; w++) {
        packed[row * (K / 4) + blk * 8 + w] = dv.getUint32(blockOff + 2 + w * 4, true)
      }
    }
  }
  return { packed, scales }
}

/** A Q8_0 weight tensor repacked for GPU in-shader dequantization. */
export interface Q8PackedWeight {
  packed: Uint32Array   // [N, K/4] u32 words (4 int8 per word)
  scales: Float32Array  // [N, K/32] f32 block scales
  N: number             // output features
  K: number             // input features
}

/** An F16 weight tensor kept as raw f16 bytes for GPU f16 storage.
 *  The shader reads array<f16> and widens to f32 at read time.
 *  Weight bandwidth: 2 bytes/element vs 4 for f32 = 2x reduction. */
export interface F16PackedWeight {
  data: Uint16Array  // [N, K] f16 values (raw bytes reinterpreted)
  N: number          // output features
  K: number          // input features
}

/** Repack F16 raw bytes into a Uint16Array for GPU upload.
 *  The GGUF F16 layout is already [N, K] row-major with 2-byte f16 values,
 *  so we just need to reinterpret the bytes as Uint16Array. */
export function repackF16(
  data: Uint8Array, N: number, K: number,
): F16PackedWeight {
  // F16 data is already in the right layout — just reinterpret as Uint16Array
  // Need to handle byteOffset alignment (Uint16Array requires 2-byte alignment)
  const u16 = new Uint16Array(data.buffer, data.byteOffset, N * K)
  // Copy to a fresh buffer to ensure alignment and ownership
  const copy = new Uint16Array(N * K)
  copy.set(u16)
  return { data: copy, N, K }
}

/** Convert IEEE 754 half-precision (f16) bits to f32. */
function f16ToF32(bits: number): number {
  const sign = (bits >> 15) & 1
  const exp = (bits >> 10) & 0x1f
  const mant = bits & 0x3ff
  if (exp === 0) {
    // Subnormal or zero
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024)
  }
  if (exp === 0x1f) {
    return mant ? NaN : (sign ? -Infinity : Infinity)
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024)
}

/** Dequantize an F16 tensor to Float32Array. */
export function dequantF16(data: Uint8Array, N: number, K: number): Float32Array {
  const out = new Float32Array(N * K)
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
  for (let i = 0; i < N * K; i++) {
    out[i] = f16ToF32(dv.getUint16(i * 2, true))
  }
  return out
}

// ─── mmproj GGUF parsing ─────────────────────────────────────────────

/** A parsed tensor from the GGUF header. */
interface GgufTensor {
  name: string
  dims: number[]
  type: number  // ggml tensor type
  off: number   // offset from data start
}

/** Parsed mmproj GGUF header. */
export interface MmprojHeader {
  meta: Record<string, unknown>
  tensors: Record<string, GgufTensor>
  dataStart: number
}

/** GGUF tensor type constants. */
const GGUF_F32 = 0
const GGUF_F16 = 1
const GGUF_Q8_0 = 8

/** Parse a GGUF v3 header from a buffer (just the header, not the weights). */
export function parseGgufHeader(buf: ArrayBuffer): MmprojHeader {
  const dv = new DataView(buf)
  const u8 = new Uint8Array(buf)
  const td = new TextDecoder()
  let pos = 0
  const need = (n: number) => {
    if (pos + n > buf.byteLength) throw new Error(`GGUF header ends at pos ${pos}, need ${n}`)
  }
  const u32 = () => { need(4); const v = dv.getUint32(pos, true); pos += 4; return v }
  const u64 = () => { need(8); const v = dv.getBigUint64(pos, true); pos += 8; return Number(v) }
  const gstr = () => {
    const n = u64()
    need(n)
    const s = td.decode(u8.subarray(pos, pos + n))
    pos += n
    return s
  }
  const value = (t: number): unknown => {
    switch (t) {
      case 0: need(1); return u8[pos++]
      case 1: need(1); return dv.getInt8(pos++)
      case 2: need(2); { const v = dv.getUint16(pos, true); pos += 2; return v }
      case 3: need(2); { const v = dv.getInt16(pos, true); pos += 2; return v }
      case 4: return u32()
      case 5: need(4); { const v = dv.getInt32(pos, true); pos += 4; return v }
      case 6: need(4); { const v = dv.getFloat32(pos, true); pos += 4; return v }
      case 7: need(1); return u8[pos++] !== 0
      case 8: return gstr()
      case 9: { const et = u32(); const n = u64(); const out: unknown[] = []; for (let i = 0; i < n; i++) out.push(value(et)); return out }
      case 10: return u64()
      case 11: need(8); { const v = dv.getBigInt64(pos, true); pos += 8; return Number(v) }
      case 12: need(8); { const v = dv.getFloat64(pos, true); pos += 8; return v }
      default: throw new Error(`GGUF: unknown value type ${t}`)
    }
  }

  need(4)
  if (td.decode(u8.subarray(0, 4)) !== 'GGUF') throw new Error('Not a GGUF file (bad magic)')
  pos = 4
  const version = u32()
  if (version !== 3) throw new Error(`Unsupported GGUF version ${version}`)
  const nTensors = u64()
  const nKv = u64()
  const meta: Record<string, unknown> = {}
  for (let i = 0; i < nKv; i++) {
    const k = gstr()
    const t = u32()
    meta[k] = value(t)
  }
  const tensors: Record<string, GgufTensor> = {}
  for (let i = 0; i < nTensors; i++) {
    const name = gstr()
    const nd = u32()
    const dims: number[] = []
    for (let d = 0; d < nd; d++) dims.push(u64())
    const type = u32()
    const off = u64()
    tensors[name] = { name, dims, type, off }
  }
  const align = Number(meta['general.alignment'] ?? 32)
  const dataStart = Math.ceil(pos / align) * align
  return { meta, tensors, dataStart }
}

/** Build a VisionConfig from the mmproj GGUF metadata. */
export function visionConfigFromMmproj(meta: Record<string, unknown>): VisionConfig {
  const P = (k: string): unknown => {
    const v = meta[`clip.${k}`]
    if (v === undefined) throw new Error(`mmproj: missing clip.${k}`)
    return v
  }
  const depth = Number(P('vision.block_count'))
  const hidden = Number(P('vision.embedding_length'))
  const inter = Number(P('vision.feed_forward_length'))
  const heads = Number(P('vision.attention.head_count'))
  const patchSize = Number(P('vision.patch_size'))
  const spatialMerge = Number(P('vision.spatial_merge_size'))
  const projDim = Number(P('vision.projection_dim'))

  // Token IDs from the language model's tokenizer (not in mmproj metadata)
  // These are the Qwen3-VL special tokens
  return {
    depth,
    hidden_size: hidden,
    intermediate_size: inter,
    num_heads: heads,
    head_dim: hidden / heads,  // 72 = 1152 / 16
    patch_size: patchSize,
    temporal_patch_size: 2,  // Qwen3-VL default
    spatial_merge_size: spatialMerge,
    out_hidden_size: projDim,  // 5120 for Bonsai-27B
    in_channels: 3,
    num_position_embeddings: 2304,  // 48×48 grid
    image_token_id: 248056,
    vision_start_token_id: 248053,
    vision_end_token_id: 248054,
  }
}

// ─── Vision weight loading ───────────────────────────────────────────

/** A loaded vision weight tensor (dequantized to F32). */
interface VisionWeight {
  name: string
  data: Float32Array
  dims: number[]
}

/** Read a tensor's raw bytes from the mmproj file.
 *  Uses HTTP range requests to fetch only the needed bytes. */
async function readTensorBytes(
  url: string,
  tensor: GgufTensor,
  dataStart: number,
  fetchRange?: (url: string, off: number, len: number) => Promise<ArrayBuffer>,
): Promise<Uint8Array> {
  // Calculate the byte size of the tensor based on type and dims
  const elems = tensor.dims.reduce((a, b) => a * b, 1)
  let bytesPerElem: number
  switch (tensor.type) {
    case GGUF_F32: bytesPerElem = 4; break
    case GGUF_F16: bytesPerElem = 2; break
    case GGUF_Q8_0: bytesPerElem = Q8_0_BLOCK_SIZE / 32; break  // 34/32 = 1.0625
    default: throw new Error(`Tensor ${tensor.name}: unsupported type ${tensor.type}`)
  }
  const byteLen = Math.ceil(elems * bytesPerElem)
  const off = dataStart + tensor.off

  if (fetchRange) {
    const buf = await fetchRange(url, off, byteLen)
    return new Uint8Array(buf)
  }
  // Default: fetch with Range header
  const res = await fetch(url, { headers: { Range: `bytes=${off}-${off + byteLen - 1}` } })
  if (!res.ok) throw new Error(`Failed to fetch tensor ${tensor.name}: HTTP ${res.status}`)
  return new Uint8Array(await res.arrayBuffer())
}

/** Load and dequantize a single tensor from the mmproj file.
 *  For Q8_0 and F16 tensors, also returns the raw bytes and type for GPU repacking. */
async function loadTensor(
  url: string,
  tensor: GgufTensor,
  dataStart: number,
  fetchRange?: (url: string, off: number, len: number) => Promise<ArrayBuffer>,
): Promise<VisionWeight & { raw?: Uint8Array, type?: number }> {
  const raw = await readTensorBytes(url, tensor, dataStart, fetchRange)
  // GGUF dims are [ne0, ne1, ...] where ne0 is innermost.
  // For a 2D weight matrix [in, out], we want [out, in] for matmul.
  // We keep the dims as-is and handle transposition in the forward pass.
  const dims = tensor.dims
  const elems = dims.reduce((a, b) => a * b, 1)

  let data: Float32Array
  switch (tensor.type) {
    case GGUF_F32:
      data = new Float32Array(raw.buffer, raw.byteOffset, elems)
      break
    case GGUF_F16:
      data = dequantF16(raw, 1, elems)
      // Keep raw bytes for GPU f16 storage (avoids re-fetching)
      return { name: tensor.name, data, dims, raw, type: GGUF_F16 }
    case GGUF_Q8_0: {
      // GGUF dims for Q8_0: [ne0, ne1] where ne0 is the innermost (fastest) dim.
      // For a weight matrix, ne0 = in_features, ne1 = out_features.
      // Q8_0 blocks are along ne0 (the innermost dimension).
      // So we dequantize as [out_features, in_features] = [ne1, ne0].
      const N = dims.length >= 2 ? dims[1] : 1  // out_features (ne1)
      const K = dims[0]  // in_features (ne0)
      data = dequantQ8_0(raw, N, K)
      // Keep raw bytes + type for GPU repacking (avoids re-fetching)
      return { name: tensor.name, data, dims, raw, type: GGUF_Q8_0 }
    }
    default:
      throw new Error(`Tensor ${tensor.name}: unsupported type ${tensor.type}`)
  }

  return { name: tensor.name, data, dims }
}

/** All loaded vision weights, organized by layer. */
export interface VisionWeights {
  // Patch embedding: two [16,16,3,1152] tensors concatenated → [1536, 1152]
  patchEmbdWeight: Float32Array  // [1536, 1152] (concatenated, transposed for matmul)
  patchEmbdBias: Float32Array   // [1152]
  // Compact patch embedding for still images: sum of temporal frames → [768, 1152]
  // For still images (frames=1), the two temporal frames are identical, so
  // W_compact = W_frame0 + W_frame1 gives the same result with half the input.
  patchEmbdWeightCompact: Float32Array  // [768, 1152]
  // Position embeddings: [2304, 1152] (1152-dim per position, 2304 positions)
  positionEmbd: Float32Array    // [2304, 1152]
  // Post layernorm
  postLnWeight: Float32Array    // [1152]
  postLnBias: Float32Array      // [1152]
  // Per-layer weights (27 layers)
  layers: VisionLayerWeights[]
  // Patch merger: mm.0 (4608→4608) + mm.2 (4608→5120)
  mergerMm0Weight: Float32Array  // [4608, 4608]
  mergerMm0Bias: Float32Array    // [4608]
  mergerMm2Weight: Float32Array  // [5120, 4608]
  mergerMm2Bias: Float32Array    // [5120]
  // Q8 packed weights for GPU in-shader dequantization (null if not Q8_0)
  q8: VisionQ8Weights | null
  // F16 packed weights for GPU f16 storage (null if not F16)
  f16: VisionF16Weights | null
}

interface VisionLayerWeights {
  ln1Weight: Float32Array   // [1152]
  ln1Bias: Float32Array     // [1152]
  qkvWeight: Float32Array   // [3456, 1152] (Q,K,V fused)
  qkvBias: Float32Array     // [3456]
  attnOutWeight: Float32Array  // [1152, 1152]
  attnOutBias: Float32Array    // [1152]
  ln2Weight: Float32Array   // [1152]
  ln2Bias: Float32Array     // [1152]
  ffnUpWeight: Float32Array // [4304, 1152]
  ffnUpBias: Float32Array   // [4304]
  ffnDownWeight: Float32Array  // [1152, 4304]
  ffnDownBias: Float32Array    // [1152]
}

/** Q8_0 packed weights for GPU in-shader dequantization.
 *  Each field is null if the weight is not Q8_0 (e.g., ffn_down is F16). */
export interface VisionQ8Weights {
  layers: Q8LayerWeights[]
  mergerMm0: Q8PackedWeight | null  // [4608, 4608]
  mergerMm2: Q8PackedWeight | null  // [5120, 4608]
}

interface Q8LayerWeights {
  qkv: Q8PackedWeight       // [3456, 1152]
  attnOut: Q8PackedWeight   // [1152, 1152]
  ffnUp: Q8PackedWeight     // [4304, 1152]
  // ffnDown is F16, not Q8_0 — stored in VisionF16Weights
}

/** F16 packed weights for GPU f16 storage (widened to f32 at read in shader).
 *  Each field is null if the weight is not F16. */
export interface VisionF16Weights {
  layers: F16LayerWeights[]
}

interface F16LayerWeights {
  ffnDown: F16PackedWeight  // [1152, 4304]
}

/** Load all vision tower weights from the mmproj GGUF file.
 *  This is the main entry point for weight loading. It:
 *  1. Fetches the GGUF header (a few MB)
 *  2. Parses tensor metadata
 *  3. Fetches + dequantizes each tensor (Q8_0 → F32, F16 → F32)
 *  4. Returns organized weight structures
 *
 *  Total download: ~600MB (the full mmproj file, read via range requests) */
export async function loadVisionWeights(
  mmprojUrl: string,
  fetchRange?: (url: string, off: number, len: number) => Promise<ArrayBuffer>,
  onProgress?: (loaded: number, total: number) => void,
): Promise<{ weights: VisionWeights; config: VisionConfig }> {
  // 1. Fetch the header (start with 1MB, grow if needed)
  const headerSize = 1 << 20  // 1MB
  const fetchFn = fetchRange ?? (async (url: string, off: number, len: number) => {
    const res = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } })
    if (!res.ok) throw new Error(`Failed to fetch mmproj header: HTTP ${res.status}`)
    return res.arrayBuffer()
  })
  let headerBuf = await fetchFn(mmprojUrl, 0, headerSize)

  // Parse header (retry with larger buffer if needed)
  let header: MmprojHeader
  try {
    header = parseGgufHeader(headerBuf)
  } catch (e) {
    // Header might be larger than 1MB — fetch more
    const bigger = await fetchFn(mmprojUrl, 0, 4 << 20)  // 4MB
    header = parseGgufHeader(bigger)
    headerBuf = bigger
  }

  const { meta, tensors: gg, dataStart } = header

  // 2. Build VisionConfig from metadata
  const config = visionConfigFromMmproj(meta)

  // 3. Bulk-fetch the entire data section in ONE request (instead of 334
  //    individual HTTP Range requests). This eliminates HTTP overhead and
  //    reduces weight loading from ~4s to ~0.5s on localhost.
  let dataEnd = 0
  for (const name in gg) {
    const t = gg[name]
    const elems = t.dims.reduce((a: number, b: number) => a * b, 1)
    let bytesPerElem: number
    switch (t.type) {
      case GGUF_F32: bytesPerElem = 4; break
      case GGUF_F16: bytesPerElem = 2; break
      case GGUF_Q8_0: bytesPerElem = Q8_0_BLOCK_SIZE / 32; break  // 34/32
      default: bytesPerElem = 4
    }
    const byteLen = Math.ceil(elems * bytesPerElem)
    const end = t.off + byteLen
    if (end > dataEnd) dataEnd = end
  }
  // Parallel fetch: split the data into 4 chunks and fetch concurrently.
  // This improves throughput by 2-3x on localhost (bypasses per-request overhead).
  const NUM_CHUNKS = 4
  const chunkSize = Math.ceil(dataEnd / NUM_CHUNKS)
  const chunks = new Array<Uint8Array>(NUM_CHUNKS)
  await Promise.all(Array.from({ length: NUM_CHUNKS }, async (_, i) => {
    const start = i * chunkSize
    const end = Math.min((i + 1) * chunkSize, dataEnd)
    if (end <= start) { chunks[i] = new Uint8Array(0); return }
    const buf = await fetchFn(mmprojUrl, dataStart + start, end - start)
    chunks[i] = new Uint8Array(buf)
  }))
  // Concatenate chunks into a single buffer
  const totalLen = chunks.reduce((a, c) => a + c.length, 0)
  const bulkData = new Uint8Array(totalLen)
  let off = 0
  for (const c of chunks) { bulkData.set(c, off); off += c.length }

  // 4. Load all tensors from the pre-fetched bulk buffer
  const depth = config.depth
  const totalTensors = depth * 12 + 8  // ~334 tensors
  let loaded = 0

  const get = async (name: string): Promise<VisionWeight> => {
    const t = gg[name]
    if (!t) throw new Error(`mmproj: tensor ${name} not found`)
    // Extract tensor bytes from the bulk buffer (no HTTP request)
    const elems = t.dims.reduce((a: number, b: number) => a * b, 1)
    let bytesPerElem: number
    switch (t.type) {
      case GGUF_F32: bytesPerElem = 4; break
      case GGUF_F16: bytesPerElem = 2; break
      case GGUF_Q8_0: bytesPerElem = Q8_0_BLOCK_SIZE / 32; break
      default: bytesPerElem = 4
    }
    const byteLen = Math.ceil(elems * bytesPerElem)
    const raw = bulkData.subarray(t.off, t.off + byteLen)
    // Dequantize based on type
    const dims = t.dims
    let data: Float32Array
    switch (t.type) {
      case GGUF_F32:
        data = new Float32Array(raw.buffer, raw.byteOffset, elems)
        break
      case GGUF_F16:
        // Skip f32 dequantization — GPU shaders use raw f16 bytes directly.
        // The f32 data is only needed for the non-F16 fallback path, which
        // is never taken when F16 weights are available.
        data = new Float32Array(0)
        return { name: t.name, data, dims, raw, type: GGUF_F16 }
      case GGUF_Q8_0: {
        // Skip f32 dequantization — GPU shaders use raw Q8 bytes directly.
        // The f32 data is only needed for the non-Q8 fallback path, which
        // is never taken when Q8 weights are available.
        data = new Float32Array(0)
        return { name: t.name, data, dims, raw, type: GGUF_Q8_0 }
      }
      default:
        throw new Error(`Tensor ${t.name}: unsupported type ${t.type}`)
    }
    loaded++
    onProgress?.(loaded, totalTensors)
    return { name: t.name, data, dims }
  }

  // Patch embedding: two weights concatenated
  const patchW0 = await get('v.patch_embd.weight')
  const patchW1 = await get('v.patch_embd.weight.1')
  const patchBias = await get('v.patch_embd.bias')
  // Each is [16, 16, 3, 1152] = 768 inputs → 1152 outputs
  // Concatenated: [1536, 1152] (2 temporal frames × 768)
  const patchEmbdWeight = new Float32Array(1536 * 1152)
  // The Conv3d weight in PyTorch is [out=1152, in=3, t=2, h=16, w=16]
  // GGUF stores it as [ne0=16, ne1=16, ne2=3, ne3=1152] where ne0 is innermost.
  // Flat layout: data[o * 768 + i] where o=out=1152, i=flattened(w,h,c)=768
  // This is already [out, in] row-major — no transposition needed.
  // patchW0 is frame 0, patchW1 is frame 1
  // Also compute compact weight: W_compact = W_frame0 + W_frame1 (for still images)
  const patchEmbdWeightCompact = new Float32Array(768 * 1152)
  for (let o = 0; o < 1152; o++) {
    for (let i = 0; i < 768; i++) {
      const w0 = patchW0.data[o * 768 + i]
      const w1 = patchW1.data[o * 768 + i]
      patchEmbdWeight[o * 1536 + i] = w0       // frame 0
      patchEmbdWeight[o * 1536 + 768 + i] = w1 // frame 1
      patchEmbdWeightCompact[o * 768 + i] = w0 + w1  // compact: sum of temporal frames
    }
  }

  // Position embeddings
  const posEmbd = await get('v.position_embd.weight')
  // GGUF dims [1152, 2304] = [ne0=1152, ne1=2304]
  // ne0=1152 is innermost (hidden_size), ne1=2304 is outermost (num_positions)
  // Flat layout: data[pos * 1152 + d] — already [num_positions, hidden_size] row-major
  // No transposition needed — copy directly
  const positionEmbd = new Float32Array(2304 * 1152)
  positionEmbd.set(posEmbd.data.subarray(0, 2304 * 1152))

  // Post layernorm
  const postLn = await get('v.post_ln.weight')
  const postLnB = await get('v.post_ln.bias')

  // Per-layer weights
  const layers: VisionLayerWeights[] = []
  const q8Layers: Q8LayerWeights[] = []
  const f16Layers: F16LayerWeights[] = []
  for (let li = 0; li < depth; li++) {
    const ln1W = await get(`v.blk.${li}.ln1.weight`)
    const ln1B = await get(`v.blk.${li}.ln1.bias`)
    const qkvW = await get(`v.blk.${li}.attn_qkv.weight`)
    const qkvB = await get(`v.blk.${li}.attn_qkv.bias`)
    const attnOutW = await get(`v.blk.${li}.attn_out.weight`)
    const attnOutB = await get(`v.blk.${li}.attn_out.bias`)
    const ln2W = await get(`v.blk.${li}.ln2.weight`)
    const ln2B = await get(`v.blk.${li}.ln2.bias`)
    const ffnUpW = await get(`v.blk.${li}.ffn_up.weight`)
    const ffnUpB = await get(`v.blk.${li}.ffn_up.bias`)
    const ffnDownW = await get(`v.blk.${li}.ffn_down.weight`)
    const ffnDownB = await get(`v.blk.${li}.ffn_down.bias`)

    // GGUF stores all weights as [ne0=in, ne1=out] → flat data is [out, in] row-major.
    // For QKV: dims [1152, 3456] → [out=3456, in=1152] ✓ (used directly)
    // For attn_out: dims [1152, 1152] → [out=1152, in=1152] ✓ (used directly)
    // For ffn_up: dims [1152, 4304] → [out=4304, in=1152] ✓ (used directly)
    // For ffn_down: dims [4304, 1152] → [out=1152, in=4304] ✓ (already correct, no transpose)
    const ffnDownData = ffnDownW.data.subarray(0, 1152 * 4304)

    layers.push({
      ln1Weight: ln1W.data,
      ln1Bias: ln1B.data,
      qkvWeight: qkvW.data,      // [3456, 1152]
      qkvBias: qkvB.data,        // [3456]
      attnOutWeight: attnOutW.data,  // [1152, 1152]
      attnOutBias: attnOutB.data,    // [1152]
      ln2Weight: ln2W.data,
      ln2Bias: ln2B.data,
      ffnUpWeight: ffnUpW.data,  // [4304, 1152]
      ffnUpBias: ffnUpB.data,    // [4304]
      ffnDownWeight: ffnDownData,  // [1152, 4304] (transposed)
      ffnDownBias: ffnDownB.data,  // [1152]
    })

    // Repack Q8_0 weights for GPU in-shader dequantization
    // Q8_0 weights: qkv, attn_out, ffn_up (ffn_down is F16)
    const qkvDims = qkvW.dims
    const qkvN = qkvDims.length >= 2 ? qkvDims[1] : 1
    const qkvK = qkvDims[0]
    const attnDims = attnOutW.dims
    const attnN = attnDims.length >= 2 ? attnDims[1] : 1
    const attnK = attnDims[0]
    const ffnUpDims = ffnUpW.dims
    const ffnUpN = ffnUpDims.length >= 2 ? ffnUpDims[1] : 1
    const ffnUpK = ffnUpDims[0]
    q8Layers.push({
      qkv: repackQ8_0(qkvW.raw!, qkvN, qkvK),
      attnOut: repackQ8_0(attnOutW.raw!, attnN, attnK),
      ffnUp: repackQ8_0(ffnUpW.raw!, ffnUpN, ffnUpK),
    })

    // Repack F16 weights for GPU f16 storage (ffn_down is F16)
    if (ffnDownW.type === GGUF_F16 && ffnDownW.raw) {
      const d = ffnDownW.dims
      const ffnDownN = d.length >= 2 ? d[1] : 1  // out=1152
      const ffnDownK = d[0]                        // in=4304
      f16Layers.push({
        ffnDown: repackF16(ffnDownW.raw, ffnDownN, ffnDownK),
      })
    }
  }

  // Patch merger
  const mm0W = await get('mm.0.weight')
  const mm0B = await get('mm.0.bias')
  const mm2W = await get('mm.2.weight')
  const mm2B = await get('mm.2.bias')

  // Repack merger weights if Q8_0
  let q8Mm0: Q8PackedWeight | null = null
  let q8Mm2: Q8PackedWeight | null = null
  if (mm0W.type === GGUF_Q8_0 && mm0W.raw) {
    const d = mm0W.dims
    q8Mm0 = repackQ8_0(mm0W.raw, d.length >= 2 ? d[1] : 1, d[0])
  }
  if (mm2W.type === GGUF_Q8_0 && mm2W.raw) {
    const d = mm2W.dims
    q8Mm2 = repackQ8_0(mm2W.raw, d.length >= 2 ? d[1] : 1, d[0])
  }

  const weights: VisionWeights = {
    patchEmbdWeight,  // [1536, 1152] = [in, out] — wait, we built it as [out, in]
    patchEmbdBias: patchBias.data,
    patchEmbdWeightCompact,  // [768, 1152] — sum of temporal frames for still images
    positionEmbd,
    postLnWeight: postLn.data,
    postLnBias: postLnB.data,
    layers,
    mergerMm0Weight: mm0W.data,  // [4608, 4608]
    mergerMm0Bias: mm0B.data,
    mergerMm2Weight: mm2W.data,  // [5120, 4608]
    mergerMm2Bias: mm2B.data,
    q8: { layers: q8Layers, mergerMm0: q8Mm0, mergerMm2: q8Mm2 },
    f16: { layers: f16Layers },
  }

  return { weights, config }
}

// ─── Vision tower state ──────────────────────────────────────────────

/** Vision tower state — loaded weights, GPU buffers, and pipelines. */
export interface VisionState {
  config: VisionConfig
  loaded: boolean
  loading: Promise<void> | null
  weights: VisionWeights | null
  // GPU buffers for vision weights (allocated on load, freed on dispose)
  buffers: Map<string, GPUBuffer>
  // Compiled pipelines for vision shaders
  pipelines: Map<string, GPURenderPipeline | GPUComputePipeline>
}

/**
 * Create an empty vision state (no weights loaded yet).
 * The mmproj pack is loaded lazily on first visionForward() call.
 */
export function createVisionState(config: VisionConfig): VisionState {
  return {
    config,
    loaded: false,
    loading: null,
    weights: null,
    buffers: new Map(),
    pipelines: new Map(),
  }
}

// ─── CPU-side vision forward pass ────────────────────────────────────

/**
 * Preprocess an image for the vision tower.
 *
 * Steps:
 * 1. Resize to the nearest multiple of (patch_size * spatial_merge_size)
 * 2. Convert to RGB float tensor [C, H, W] with values normalized by mean/std
 * 3. Reshape to patches [num_patches, temporal * patch * patch * C]
 *
 * This runs on CPU (one-time per image, not per layer).
 */
/**
 * Qwen3-VL smart_resize: rescale image so both dims are divisible by `factor`
 * (patch_size * spatial_merge_size) and total pixels ≤ maxPixels.
 * Mirrors qwen_vl_utils/vision_process.py:smart_resize.
 */
export function smartResize(
  height: number,
  width: number,
  factor: number,
  minPixels = 4 * factor * factor,
  maxPixels = 256 * factor * factor,
): [number, number] {
  const roundBy = (n: number, f: number) => Math.round(n / f) * f
  const floorBy = (n: number, f: number) => Math.floor(n / f) * f
  const ceilBy = (n: number, f: number) => Math.ceil(n / f) * f

  let hBar = Math.max(factor, roundBy(height, factor))
  let wBar = Math.max(factor, roundBy(width, factor))
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels)
    hBar = Math.max(factor, floorBy(height / beta, factor))
    wBar = Math.max(factor, floorBy(width / beta, factor))
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width))
    hBar = Math.max(factor, ceilBy(height * beta, factor))
    wBar = Math.max(factor, ceilBy(width * beta, factor))
  }
  return [hBar, wBar]
}

/**
 * Bicubic resize of an RGB Float32Array image (matches PIL BICUBIC).
 * Uses Catmull-Rom cubic interpolation (a=-0.75, matching PIL's default).
 * Input:  rgb [3 * srcH * srcW], values in [0,1]
 * Output: [3 * dstH * dstW]
 */
function bicubicResize(
  rgb: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Float32Array {
  const out = new Float32Array(3 * dstW * dstH)
  // PIL BICUBIC uses scale = src / dst (not (src-1)/(dst-1) like bilinear)
  const xScale = srcW / dstW
  const yScale = srcH / dstH

  // Catmull-Rom cubic kernel with a=-0.5 (matches PIL BICUBIC / antialias)
  // PIL uses a=-0.5 for its bicubic filter, NOT a=-0.75 (Catmull-Rom).
  // This difference is critical for text recognition accuracy in VLMs.
  // Uses |t| — the kernel is symmetric (even function)
  const cubicKernel = (t: number): number => {
    const a = -0.5
    const at = Math.abs(t)
    const t2 = at * at
    const t3 = t2 * at
    if (at < 1) return (a + 2) * t3 - (a + 3) * t2 + 1
    if (at < 2) return a * t3 - 5 * a * t2 + 8 * a * at - 4 * a
    return 0
  }

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

  for (let dy = 0; dy < dstH; dy++) {
    const sy = (dy + 0.5) * yScale - 0.5
    const y0 = Math.floor(sy)
    const fy = sy - y0
    for (let dx = 0; dx < dstW; dx++) {
      const sx = (dx + 0.5) * xScale - 0.5
      const x0 = Math.floor(sx)
      const fx = sx - x0

      // Precompute x weights (4 taps)
      const xw = [
        cubicKernel(-1 - fx),
        cubicKernel(-fx),
        cubicKernel(1 - fx),
        cubicKernel(2 - fx),
      ]
      // Precompute y weights (4 taps)
      const yw = [
        cubicKernel(-1 - fy),
        cubicKernel(-fy),
        cubicKernel(1 - fy),
        cubicKernel(2 - fy),
      ]

      for (let c = 0; c < 3; c++) {
        let acc = 0
        for (let j = 0; j < 4; j++) {
          const yy = clamp(y0 + j - 1, 0, srcH - 1)
          let rowAcc = 0
          for (let i = 0; i < 4; i++) {
            const xx = clamp(x0 + i - 1, 0, srcW - 1)
            rowAcc += rgb[(yy * srcW + xx) * 3 + c] * xw[i]
          }
          acc += rowAcc * yw[j]
        }
        out[(dy * dstW + dx) * 3 + c] = clamp(acc, 0, 1)
      }
    }
  }
  return out
}

export function preprocessImage(
  image: ImageInput,
  config: VisionConfig,
  maxPixels?: number,
): { patches: Float32Array; numPatches: number; gridThw: [number, number, number]; compact: boolean } {
  let { width, height, rgb, frames = 1 } = image
  const { patch_size, temporal_patch_size, spatial_merge_size, in_channels } = config

  // Qwen3-VL smart_resize: downscale large images to limit token count.
  // factor = patch_size * spatial_merge_size (e.g. 16 * 2 = 32)
  // Default maxPixels = 256 tokens × factor² ≈ 262K pixels (→ ~256 merged patches)
  const factor = patch_size * spatial_merge_size
  const mp = maxPixels ?? 256 * factor * factor
  const [resizedH, resizedW] = smartResize(height, width, factor, 4 * factor * factor, mp)
  if (resizedH !== height || resizedW !== width) {
    rgb = bicubicResize(rgb, width, height, resizedW, resizedH)
    width = resizedW
    height = resizedH
  }

  // Image normalization: mean=0.5, std=0.5 (from clip.vision.image_mean/std)
  const mean = 0.5
  const std = 0.5

  // Dimensions are now exact multiples of mergedPatch (guaranteed by smartResize)
  const numTemporal = Math.max(1, Math.ceil(frames / temporal_patch_size))
  const numHeightPatches = height / patch_size
  const numWidthPatches = width / patch_size
  const numPatches = numTemporal * numHeightPatches * numWidthPatches

  // Compact patch embedding for still images: skip temporal duplication.
  // For still images (frames=1), the two temporal frames are identical, so
  // we can use 768-dim patches (no temporal) with the compact weight
  // (W_frame0 + W_frame1) instead of 1536-dim patches with the full weight.
  const isStillImage = frames === 1
  const compact = isStillImage
  const patchDim = compact
    ? in_channels * patch_size * patch_size                    // 768
    : in_channels * temporal_patch_size * patch_size * patch_size  // 1536

  // CRITICAL: Patches are arranged in [t, h_blk, w_blk, h_intra, w_intra] order
  // (not standard [t, h, w] order). This matches the reference image processor
  // which transposes to (grid_t, h_blk, w_blk, h_intra, w_intra, C, T, P, P)
  // before flattening. This order is required so that:
  // 1. Position embeddings (also in this order) are added to the correct patches
  // 2. RoPE positions (also in this order) encode correct spatial relationships
  // 3. The spatial merger groups adjacent 2×2 patches correctly
  const mergeSize = spatial_merge_size
  const numHeightBlocks = numHeightPatches / mergeSize
  const numWidthBlocks = numWidthPatches / mergeSize

  const patches = new Float32Array(numPatches * patchDim)

  for (let t = 0; t < numTemporal; t++) {
    for (let hb = 0; hb < numHeightBlocks; hb++) {
      for (let wb = 0; wb < numWidthBlocks; wb++) {
        for (let ih = 0; ih < mergeSize; ih++) {
          for (let iw = 0; iw < mergeSize; iw++) {
            const h = hb * mergeSize + ih
            const w = wb * mergeSize + iw
            // Patch index in [t, h_blk, w_blk, h_intra, w_intra] order
            const patchIdx = (((t * numHeightBlocks + hb) * numWidthBlocks + wb) * mergeSize + ih) * mergeSize + iw
            const patchBase = patchIdx * patchDim

            let dimIdx = 0
            for (let c = 0; c < in_channels; c++) {
              const ttMax = compact ? 1 : temporal_patch_size
              for (let tt = 0; tt < ttMax; tt++) {
                for (let ph = 0; ph < patch_size; ph++) {
                  for (let pw = 0; pw < patch_size; pw++) {
                    const frame = t * temporal_patch_size + tt
                    const py = h * patch_size + ph
                    const px = w * patch_size + pw
                    if (py < height && px < width && frame < frames) {
                      const srcIdx = (frame * height * width + py * width + px) * in_channels + c
                      patches[patchBase + dimIdx] = (rgb[srcIdx] - mean) / std
                    }
                    dimIdx++
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  const gridThw: [number, number, number] = [numTemporal, numHeightPatches, numWidthPatches]
  return { patches, numPatches, gridThw, compact }
}

/**
 * Compute interpolated position embeddings for the vision tower.
 * Bilinear interpolation from 48×48 learned grid to actual patch grid,
 * then rearranged to [t, h_blk, w_blk, h_intra, w_intra] order to match
 * the patch ordering from preprocessImage.
 * Runs on CPU (one-time per image, not per layer).
 */
export function computeVisionPosEmbed(
  gridThw: [number, number, number],
  config: VisionConfig,
  posEmbedTable: Float32Array,  // [2304, 1152] = [num_positions, hidden_size]
): Float32Array {
  const [temporal, height, width] = gridThw
  const { num_position_embeddings, hidden_size, spatial_merge_size } = config
  const gridPerSide = Math.sqrt(num_position_embeddings)  // 48
  const mergeSize = spatial_merge_size

  // Position embeddings are added BEFORE the spatial merge, so we need
  // one per patch (not per merged patch)
  const numPositions = temporal * height * width
  const posEmbeds = new Float32Array(numPositions * hidden_size)

  // First, compute interpolated pos embeds in standard [h, w] order
  const hwPosEmbeds = new Float32Array(height * width * hidden_size)
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const hwIdx = h * width + w
      const hwBase = hwIdx * hidden_size

      // Bilinear interpolation from 48×48 grid to height × width grid
      // Uses linspace(0, gridPerSide-1, h) which is equivalent to
      // h * (gridPerSide - 1) / max(height - 1, 1) (align_corners=True)
      const srcH = h * (gridPerSide - 1) / Math.max(height - 1, 1)
      const srcW = w * (gridPerSide - 1) / Math.max(width - 1, 1)
      const h0 = Math.floor(srcH)
      const w0 = Math.floor(srcW)
      const h1 = Math.min(h0 + 1, gridPerSide - 1)
      const w1 = Math.min(w0 + 1, gridPerSide - 1)
      const dh = srcH - h0
      const dw = srcW - w0

      for (let d = 0; d < hidden_size; d++) {
        const v00 = posEmbedTable[(h0 * gridPerSide + w0) * hidden_size + d]
        const v01 = posEmbedTable[(h0 * gridPerSide + w1) * hidden_size + d]
        const v10 = posEmbedTable[(h1 * gridPerSide + w0) * hidden_size + d]
        const v11 = posEmbedTable[(h1 * gridPerSide + w1) * hidden_size + d]
        const v0 = v00 * (1 - dw) + v01 * dw
        const v1 = v10 * (1 - dw) + v11 * dw
        hwPosEmbeds[hwBase + d] = v0 * (1 - dh) + v1 * dh
      }
    }
  }

  // Then rearrange to [t, h_blk, w_blk, h_intra, w_intra] order
  // (matching the reference's permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
  const numHeightBlocks = height / mergeSize
  const numWidthBlocks = width / mergeSize
  for (let t = 0; t < temporal; t++) {
    for (let hb = 0; hb < numHeightBlocks; hb++) {
      for (let wb = 0; wb < numWidthBlocks; wb++) {
        for (let ih = 0; ih < mergeSize; ih++) {
          for (let iw = 0; iw < mergeSize; iw++) {
            const h = hb * mergeSize + ih
            const w = wb * mergeSize + iw
            const srcIdx = h * width + w
            const dstIdx = (((t * numHeightBlocks + hb) * numWidthBlocks + wb) * mergeSize + ih) * mergeSize + iw
            for (let d = 0; d < hidden_size; d++) {
              posEmbeds[dstIdx * hidden_size + d] = hwPosEmbeds[srcIdx * hidden_size + d]
            }
          }
        }
      }
    }
  }

  return posEmbeds
}

/**
 * Compute cu_seqlens for batched image attention.
 * Mirrors HuggingFace's `get_vision_cu_seqlens` with merge_temporal=False
 * (the qwen3_vl default): one attention segment per frame, not per image.
 *
 * E.g., for 2 images each with t=1, h=14, w=14: [0, 196, 392]
 * For 1 video with t=3, h=14, w=14: [0, 196, 392, 588]  (3 segments)
 */
export function computeCuSeqlens(
  images: { gridThw: [number, number, number] }[],
  _config?: VisionConfig,
): Uint32Array {
  // Count total segments: one per frame per image
  let numSegments = 0
  for (const img of images) numSegments += img.gridThw[0]
  const cuSeqlens = new Uint32Array(numSegments + 1)
  cuSeqlens[0] = 0
  let seg = 0
  for (const img of images) {
    const [t, h, w] = img.gridThw
    const framePatches = h * w
    for (let f = 0; f < t; f++) {
      seg++
      cuSeqlens[seg] = cuSeqlens[seg - 1] + framePatches
    }
  }
  return cuSeqlens
}

/**
 * Compute the number of image tokens to insert into the token sequence.
 * Each merged patch becomes one image_token_id in the token stream.
 */
export function numImageTokens(
  gridThw: [number, number, number],
  config: VisionConfig,
): number {
  const [t, h, w] = gridThw
  const { spatial_merge_size } = config
  return t * (h / spatial_merge_size) * (w / spatial_merge_size)
}

/**
 * Compute 2D position IDs for the vision rotary embedding.
 *
 * Patches are in [t, h_blk, w_blk, h_intra, w_intra] order (matching the
 * reference image processor's transpose). Position IDs must be in the same
 * order: for each block (hb, wb), for each intra (ih, iw), the position is
 * (hb * mergeSize + ih, wb * mergeSize + iw).
 *
 * Returns position_ids: Float32Array [total_patches, 2] (h_pos, w_pos per patch)
 */
export function computeVisionPositionIds(
  gridThw: [number, number, number],
  spatialMergeSize: number,
): Float32Array {
  const [t, h, w] = gridThw
  const mergeSize = spatialMergeSize
  const numHeightBlocks = h / mergeSize
  const numWidthBlocks = w / mergeSize
  const patchesPerFrame = h * w
  const posIds = new Float32Array(t * patchesPerFrame * 2)

  for (let frame = 0; frame < t; frame++) {
    const frameOff = frame * patchesPerFrame * 2
    let idx = 0
    for (let hb = 0; hb < numHeightBlocks; hb++) {
      for (let wb = 0; wb < numWidthBlocks; wb++) {
        for (let ih = 0; ih < mergeSize; ih++) {
          for (let iw = 0; iw < mergeSize; iw++) {
            posIds[frameOff + idx * 2] = hb * mergeSize + ih
            posIds[frameOff + idx * 2 + 1] = wb * mergeSize + iw
            idx++
          }
        }
      }
    }
  }
  return posIds
}

/**
 * Compute cos/sin caches for the vision 2D RoPE.
 *
 * Mirrors `Qwen3VLVisionRotaryEmbedding`:
 *  - dim = head_dim // 2 (partial rotary, 50%)
 *  - inv_freq[i] = 1.0 / (theta ^ (2*i / dim)), i = 0..dim/2-1
 *  - freqs = [h * inv_freq, w * inv_freq] → (head_dim//2) values per patch
 *  - emb = cat(freqs, freqs) → head_dim values
 *  - cos = emb.cos(), sin = emb.sin()
 *
 * Returns { cos, sin } each Float32Array [numPatches, head_dim]
 */
export function computeVisionRoPE(
  gridThw: [number, number, number],
  spatialMergeSize: number,
  headDim: number,
  theta: number = 10000.0,
): { cos: Float32Array; sin: Float32Array } {
  const rotaryDim = headDim // 2  // partial rotary: 50% of head_dim
  const halfRotary = rotaryDim // 2  // = head_dim // 4
  const [t, h, w] = gridThw
  const numPatches = t * h * w

  // inv_freq: halfRotary elements
  const invFreq = new Float32Array(halfRotary)
  for (let i = 0; i < halfRotary; i++) {
    invFreq[i] = 1.0 / Math.pow(theta, (2 * i) / rotaryDim)
  }

  const posIds = computeVisionPositionIds(gridThw, spatialMergeSize)

  const cos = new Float32Array(numPatches * headDim)
  const sin = new Float32Array(numPatches * headDim)

  for (let p = 0; p < numPatches; p++) {
    const hPos = posIds[p * 2]
    const wPos = posIds[p * 2 + 1]

    // freqs: [hPos * inv_freq (halfRotary), wPos * inv_freq (halfRotary)] = rotaryDim values
    // emb = cat(freqs, freqs) = headDim values
    // cos/sin = emb.cos()/sin()
    for (let d = 0; d < headDim; d++) {
      let freq: number
      if (d < rotaryDim) {
        // First half: freqs[d] = (d < halfRotary) ? hPos * invFreq[d] : wPos * invFreq[d - halfRotary]
        if (d < halfRotary) {
          freq = hPos * invFreq[d]
        } else {
          freq = wPos * invFreq[d - halfRotary]
        }
      } else {
        // Second half: repeat of first half
        const d2 = d - rotaryDim
        if (d2 < halfRotary) {
          freq = hPos * invFreq[d2]
        } else {
          freq = wPos * invFreq[d2 - halfRotary]
        }
      }
      cos[p * headDim + d] = Math.cos(freq)
      sin[p * headDim + d] = Math.sin(freq)
    }
  }

  return { cos, sin }
}

// ─── CPU-side matmul + layer norm + GELU helpers ─────────────────────

/** Matrix multiply: C[M,N] = A[M,K] × B[N,K]^T + bias[N]
 *  B is stored as [N, K] (row-major, output × input). */
function matmul(a: Float32Array, b: Float32Array, bias: Float32Array, M: number, N: number, K: number): Float32Array {
  const out = new Float32Array(M * N)
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = bias[n]
      for (let k = 0; k < K; k++) {
        sum += a[m * K + k] * b[n * K + k]
      }
      out[m * N + n] = sum
    }
  }
  return out
}

/** LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta */
function layerNorm(x: Float32Array, gamma: Float32Array, beta: Float32Array, M: number, D: number, eps: number): Float32Array {
  const out = new Float32Array(M * D)
  for (let m = 0; m < M; m++) {
    const base = m * D
    let mean = 0
    for (let d = 0; d < D; d++) mean += x[base + d]
    mean /= D
    let variance = 0
    for (let d = 0; d < D; d++) { const v = x[base + d] - mean; variance += v * v }
    variance /= D
    const invStd = 1 / Math.sqrt(variance + eps)
    for (let d = 0; d < D; d++) {
      out[base + d] = (x[base + d] - mean) * invStd * gamma[d] + beta[d]
    }
  }
  return out
}

/** GELU (pytorch_tanh approximation). */
function gelu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length)
  const coef = 0.7978845608028654  // sqrt(2/pi)
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    out[i] = 0.5 * v * (1 + Math.tanh(coef * (v + 0.044715 * v * v * v)))
  }
  return out
}

/** Add residual: y = a + b */
function addResidual(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
  return out
}

/** Bidirectional full self-attention with 2D RoPE, per-frame segments.
 *  Q, K, V: [total_seq, heads, head_dim]
 *  cos, sin: [total_seq, head_dim] (precomputed by computeVisionRoPE)
 *  cuSeqlens: cumulative segment boundaries [numSegs+1] (e.g. [0, 196, 392] for 2 frames)
 *  No causal mask — every patch in a segment attends to every other patch in the same segment. */
function visionAttention(
  q: Float32Array, k: Float32Array, v: Float32Array,
  totalSeq: number, numHeads: number, headDim: number,
  cos?: Float32Array, sin?: Float32Array,
  cuSeqlens?: Uint32Array,
): Float32Array {
  const scale = 1 / Math.sqrt(headDim)
  const rotaryDim = cos ? headDim / 2 : 0
  const out = new Float32Array(totalSeq * numHeads * headDim)

  // Default: single segment covering all patches
  const seglens = cuSeqlens ?? new Uint32Array([0, totalSeq])
  const numSegs = seglens.length - 1

  // Apply RoPE to Q and K (in-place on copies)
  let qRoped = q, kRoped = k
  if (cos && sin) {
    qRoped = new Float32Array(q.length)
    kRoped = new Float32Array(k.length)
    for (let p = 0; p < totalSeq; p++) {
      const ropeBase = p * headDim
      for (let h = 0; h < numHeads; h++) {
        const qBase = p * numHeads * headDim + h * headDim
        const kBase = p * numHeads * headDim + h * headDim
        for (let d = 0; d < headDim; d++) {
          const c = cos[ropeBase + d]
          const s = sin[ropeBase + d]
          const rotated_q = d < rotaryDim ? -q[qBase + d + rotaryDim] : q[qBase + d - rotaryDim]
          const rotated_k = d < rotaryDim ? -k[kBase + d + rotaryDim] : k[kBase + d - rotaryDim]
          qRoped[qBase + d] = q[qBase + d] * c + rotated_q * s
          kRoped[kBase + d] = k[kBase + d] * c + rotated_k * s
        }
      }
    }
  }

  for (let seg = 0; seg < numSegs; seg++) {
    const segStart = seglens[seg]
    const segEnd = seglens[seg + 1]
    const segLen = segEnd - segStart
    for (let h = 0; h < numHeads; h++) {
      for (let qi = 0; qi < segLen; qi++) {
        const qIdx = segStart + qi
        const qBase = qIdx * numHeads * headDim + h * headDim
        // Compute attention scores for this query against all keys in this segment
        const scores = new Float32Array(segLen)
        let maxScore = -Infinity
        for (let ki = 0; ki < segLen; ki++) {
          const kIdx = segStart + ki
          const kBase = kIdx * numHeads * headDim + h * headDim
          let dot = 0
          for (let d = 0; d < headDim; d++) dot += qRoped[qBase + d] * kRoped[kBase + d]
          scores[ki] = dot * scale
          if (scores[ki] > maxScore) maxScore = scores[ki]
        }
        // Softmax
        let sumExp = 0
        for (let ki = 0; ki < segLen; ki++) {
          scores[ki] = Math.exp(scores[ki] - maxScore)
          sumExp += scores[ki]
        }
        // Weighted sum of values
        const outBase = qBase
        for (let d = 0; d < headDim; d++) out[outBase + d] = 0
        for (let ki = 0; ki < segLen; ki++) {
          const kIdx = segStart + ki
          const vBase = kIdx * numHeads * headDim + h * headDim
          const w = scores[ki] / sumExp
          for (let d = 0; d < headDim; d++) out[outBase + d] += w * v[vBase + d]
        }
      }
    }
  }
  return out
}

// ─── Full CPU-side vision forward pass ───────────────────────────────

/**
 * Run the full vision tower forward pass on CPU.
 *
 * This is a reference implementation that runs entirely on CPU.
 * It's correct but slow — for production, the GPU shaders should be used.
 * The GPU implementation will use the WGSL shaders in shaders/vision_*.wgsl.
 *
 * Pipeline:
 * 1. Patch embed: linear(1536→1152) + bias
 * 2. Add position embeddings (bilinear interpolated)
 * 3. 27× VisionBlock:
 *    a. LayerNorm1 → QKV proj → bidirectional attention → Out proj → residual
 *    b. LayerNorm2 → FFN up → GELU → FFN down → residual
 * 4. Post layernorm
 * 5. Patch merger: pixel shuffle 2x2 → linear(4608→4608) → GELU → linear(4608→5120)
 */
export function visionForwardCpu(
  images: ImageInput[],
  weights: VisionWeights,
  config: VisionConfig,
): VisionForwardResult {
  const t0 = performance.now()
  const { depth, hidden_size, intermediate_size, num_heads, head_dim,
    patch_size, temporal_patch_size, spatial_merge_size, out_hidden_size,
    in_channels } = config

  const eps = 1e-6

  // Process each image
  const allImageEmbeds: Float32Array[] = []
  const allNumMerged: number[] = []

  for (const image of images) {
    // 1. Preprocess → patches
    const { patches, numPatches, gridThw, compact } = preprocessImage(image, config)

    // 2. Patch embedding: linear(in_dim→1152) + bias
    // For still images: compact 768-dim patches with compact weight (sum of temporal frames)
    // For video: full 1536-dim patches with full weight
    const patchWeight = compact ? weights.patchEmbdWeightCompact : weights.patchEmbdWeight
    const patchInDim = compact
      ? in_channels * patch_size * patch_size                    // 768
      : in_channels * temporal_patch_size * patch_size * patch_size  // 1536
    let hidden_states = matmul(patches, patchWeight, weights.patchEmbdBias,
      numPatches, hidden_size, patchInDim)

    // 3. Add position embeddings
    const posEmbeds = computeVisionPosEmbed(gridThw, config, weights.positionEmbd)
    hidden_states = addResidual(hidden_states, posEmbeds)

    // 3b. Compute 2D RoPE cos/sin caches (one-time per image)
    const { cos: ropeCos, sin: ropeSin } = computeVisionRoPE(gridThw, spatial_merge_size, head_dim)

    // 3c. Compute cu_seqlens: one attention segment per frame
    const cuSeqlens = computeCuSeqlens([{ gridThw }], config)

    // 4. 27 transformer blocks
    for (let li = 0; li < depth; li++) {
      const lw = weights.layers[li]

      // 4a. Attention block
      const normed = layerNorm(hidden_states, lw.ln1Weight, lw.ln1Bias, numPatches, hidden_size, eps)
      // QKV projection: [numPatches, 1152] → [numPatches, 3456]
      const qkv = matmul(normed, lw.qkvWeight, lw.qkvBias, numPatches, 3 * hidden_size, hidden_size)
      // Split Q, K, V: each [numPatches, 1152] = [numPatches, heads, head_dim]
      const q = new Float32Array(numPatches * hidden_size)
      const k = new Float32Array(numPatches * hidden_size)
      const v = new Float32Array(numPatches * hidden_size)
      for (let p = 0; p < numPatches; p++) {
        for (let d = 0; d < hidden_size; d++) {
          q[p * hidden_size + d] = qkv[p * 3 * hidden_size + d]
          k[p * hidden_size + d] = qkv[p * 3 * hidden_size + hidden_size + d]
          v[p * hidden_size + d] = qkv[p * 3 * hidden_size + 2 * hidden_size + d]
        }
      }
      // Bidirectional attention with 2D RoPE (per-frame segments)
      const attnOut = visionAttention(q, k, v, numPatches, num_heads, head_dim, ropeCos, ropeSin, cuSeqlens)
      // Output projection: [numPatches, 1152] → [numPatches, 1152]
      const attnProj = matmul(attnOut, lw.attnOutWeight, lw.attnOutBias, numPatches, hidden_size, hidden_size)
      hidden_states = addResidual(hidden_states, attnProj)

      // 4b. MLP block
      const normed2 = layerNorm(hidden_states, lw.ln2Weight, lw.ln2Bias, numPatches, hidden_size, eps)
      // FFN up: [numPatches, 1152] → [numPatches, 4304]
      const up = matmul(normed2, lw.ffnUpWeight, lw.ffnUpBias, numPatches, intermediate_size, hidden_size)
      // GELU
      const act = gelu(up)
      // FFN down: [numPatches, 4304] → [numPatches, 1152]
      const down = matmul(act, lw.ffnDownWeight, lw.ffnDownBias, numPatches, hidden_size, intermediate_size)
      hidden_states = addResidual(hidden_states, down)
    }

    // 5. Post layernorm
    hidden_states = layerNorm(hidden_states, weights.postLnWeight, weights.postLnBias, numPatches, hidden_size, eps)

    // 6. Patch merger: pixel shuffle 2x2 → 4608 → linear → GELU → linear → 5120
    const [t, h, w] = gridThw
    const mergedH = h / spatial_merge_size
    const mergedW = w / spatial_merge_size
    const numMerged = t * mergedH * mergedW
    const mergedDim = hidden_size * spatial_merge_size * spatial_merge_size  // 4608

    // Pixel shuffle: merge 2x2 spatial patches into one vector.
    // Patches are in [t, h_blk, w_blk, h_intra, w_intra] order, so consecutive
    // groups of 4 patches form one 2x2 merge block. The intra-block order is
    // (ih=0,iw=0), (ih=0,iw=1), (ih=1,iw=0), (ih=1,iw=1) = (sh,sw) order.
    const mergeUnit = spatial_merge_size * spatial_merge_size  // 4
    const shuffled = new Float32Array(numMerged * mergedDim)
    for (let mi = 0; mi < numMerged; mi++) {
      for (let s = 0; s < mergeUnit; s++) {
        const origIdx = mi * mergeUnit + s
        for (let d = 0; d < hidden_size; d++) {
          shuffled[mi * mergedDim + s * hidden_size + d] =
            hidden_states[origIdx * hidden_size + d]
        }
      }
    }

    // mm.0: linear(4608→4608) + bias
    const mm0Out = matmul(shuffled, weights.mergerMm0Weight, weights.mergerMm0Bias,
      numMerged, mergedDim, mergedDim)
    // GELU
    const mm0Act = gelu(mm0Out)
    // mm.2: linear(4608→5120) + bias
    const imageEmbeds = matmul(mm0Act, weights.mergerMm2Weight, weights.mergerMm2Bias,
      numMerged, out_hidden_size, mergedDim)

    allImageEmbeds.push(imageEmbeds)
    allNumMerged.push(numMerged)
  }

  // Concatenate all image embeddings
  const totalMerged = allNumMerged.reduce((a, b) => a + b, 0)
  const imageEmbeds = new Float32Array(totalMerged * out_hidden_size)
  let offset = 0
  for (let i = 0; i < allImageEmbeds.length; i++) {
    imageEmbeds.set(allImageEmbeds[i], offset)
    offset += allImageEmbeds[i].length
  }

  const elapsedMs = performance.now() - t0
  return {
    imageEmbeds,
    numPatches: totalMerged,
    deepstackFeatures: [],  // No DeepStack — is_deepstack_layers is all false
    elapsedMs,
  }
}

/**
 * Vision forward pass entry point.
 *
 * Currently runs on CPU (visionForwardCpu). The GPU implementation will
 * use the WGSL shaders in shaders/vision_*.wgsl once the vision weight
 * loading to GPU buffers is implemented.
 */
export async function visionForward(
  images: ImageInput[],
  state: VisionState,
  _device: GPUDevice,
): Promise<VisionForwardResult> {
  if (!state.weights) {
    throw new Error('visionForward: weights not loaded. Call loadVisionWeights first.')
  }
  // CPU reference implementation — correct but slow.
  // GPU implementation will use the compiled WGSL shaders.
  return visionForwardCpu(images, state.weights, state.config)
}
