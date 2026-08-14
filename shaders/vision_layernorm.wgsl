// Vision LayerNorm — the vision tower uses LayerNorm (with weight + bias), NOT RMSNorm.
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// eps = 1e-6 (from clip.vision.layer_norm_epsilon)
//
// Uses @workgroup_size(64) with a single-thread two-pass reduction.
// One workgroup per row — 64 threads are available but only thread 0 does the
// reduction (the workgroup size is set to 64 for GPU occupancy efficiency;
// the extra threads are idle but the scheduler handles this better than wg=1).
struct Params {
  R: u32,    // number of rows (seq_len)
  D: u32,    // hidden_size (1152)
  eps: f32,  // 1e-6 for vision
  _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;  // ln weight
@group(0) @binding(3) var<storage, read> beta: array<f32>;   // ln bias
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;

  // First pass: compute mean
  var sum = 0.0;
  for (var i = 0u; i < p.D; i = i + 1u) {
    sum = sum + x[base + i];
  }
  let mean = sum / f32(p.D);

  // Second pass: compute variance (stable: sum((x-mean)^2) / D)
  var sum_sq = 0.0;
  for (var i = 0u; i < p.D; i = i + 1u) {
    let diff = x[base + i] - mean;
    sum_sq = sum_sq + diff * diff;
  }
  let variance = sum_sq / f32(p.D);
  let inv_std = inverseSqrt(variance + p.eps);

  // Third pass: normalize and apply affine
  for (var i = 0u; i < p.D; i = i + 1u) {
    y[base + i] = (x[base + i] - mean) * inv_std * gamma[i] + beta[i];
  }
}
