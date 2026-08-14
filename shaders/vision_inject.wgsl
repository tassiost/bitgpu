// Replace image_token_id positions in the token embedding buffer with vision embeddings.
// One workgroup per image token position; 64 threads split the hidden_dim (5120 → 80 per thread).
struct Params {
  hidden_dim: u32,         // LLM hidden dim (5120) — equals vision projection_dim
  num_image_tokens: u32,   // total image token positions to overwrite
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> positions: array<u32>;           // [num_image_tokens] seq positions
@group(0) @binding(2) var<storage, read> vision_embeds: array<f32>;       // [num_image_tokens, hidden_dim]
@group(0) @binding(3) var<storage, read_write> token_embeds: array<f32>;  // [seq_len, hidden_dim]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let img_idx = wg.x;
  if (img_idx >= p.num_image_tokens) { return; }

  let seq_pos = positions[img_idx];
  let base_out = seq_pos * p.hidden_dim;
  let base_in = img_idx * p.hidden_dim;

  for (var d = lid.x; d < p.hidden_dim; d = d + 64u) {
    token_embeds[base_out + d] = vision_embeds[base_in + d];
  }
}
