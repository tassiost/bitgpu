// Q8_0 dequantization for the vision tower weights.
// Q8_0 is a standard ggml format: blocks of 32 elements, each block has:
//   - 1 f16 scale (2 bytes)
//   - 32 int8 values (32 bytes)
//   Total: 34 bytes per 32 elements = 1.0625 bytes per element
//
// Dequantization: value = f32(int8_value) * f32(scale)
//
// The mmproj file uses Q8_0 for most vision weights (attn_qkv, attn_out,
// ffn_up, mm.0, mm.2). The ffn_down weights are F16 (not Q8_0).
//
// This shader dequantizes Q8_0 packed data into f32 tensors that the
// standard matmul kernels can consume. One invocation per output element.

struct Params {
  N: u32,           // output rows
  K: u32,           // output cols (full, not packed)
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
// Q8_0 layout: [N, K/32 blocks], each block = 2-byte f16 scale + 32 int8 values = 34 bytes
@group(0) @binding(1) var<storage, read> packed: array<u32>;  // raw Q8_0 bytes as u32 words
@group(0) @binding(2) var<storage, read_write> out: array<f32>;  // [N, K] dequantized

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let total = p.N * p.K;
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= total) { return; }

  let row = idx / p.K;
  let col = idx % p.K;
  let block_idx = col / 32u;     // which 32-element block
  let within = col % 32u;        // offset within the block

  // Q8_0 block layout (34 bytes per block):
  //   bytes 0-1: f16 scale (little-endian)
  //   bytes 2-33: 32 × int8 values
  // We read the raw bytes from the packed u32 array.
  // Byte offset for this block: row * (K/32) * 34 + block_idx * 34
  let blocks_per_row = p.K / 32u;
  let block_byte_off = row * blocks_per_row * 34u + block_idx * 34u;

  // Read f16 scale (first 2 bytes of the block)
  let scale_byte_off = block_byte_off;
  let scale_u32_idx = scale_byte_off / 4u;
  let scale_byte_shift = (scale_byte_off % 4u) * 8u;
  let scale_raw = (packed[scale_u32_idx] >> scale_byte_shift) & 0xFFFFu;
  // Convert f16 to f32
  let scale_f16 = unpack2x16float(vec2<u32>(scale_raw, 0u)).x;

  // Read int8 value (byte 2 + within)
  let val_byte_off = block_byte_off + 2u + within;
  let val_u32_idx = val_byte_off / 4u;
  let val_byte_shift = (val_byte_off % 4u) * 8u;
  let val_u8 = (packed[val_u32_idx] >> val_byte_shift) & 0xFFu;
  // Convert to signed int8: if bit 7 is set, subtract 256
  let val_i8 = f32(val_u8) - f32(select(0u, 256u, val_u8 >= 128u));

  out[idx] = val_i8 * scale_f16;
}
