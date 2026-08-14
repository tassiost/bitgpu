// Vision LayerNorm — the vision tower uses LayerNorm (with weight + bias), NOT RMSNorm.
// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// eps = 1e-6 (from clip.vision.layer_norm_epsilon)
//
// Uses @workgroup_size(64) with a single-thread reduction.
// One workgroup per row — all 64 threads execute the same code in SIMD lockstep
// (no divergence, no barriers), which is faster on Apple Silicon than a parallel
// tree reduction that requires 12+ workgroupBarriers per row.
//
// Testing showed parallel tree reduction was 22% SLOWER due to barrier overhead:
// - Old (single-thread, no barriers): 33.9s vision tower
// - New (tree reduction, 12 barriers/row): 41.6s vision tower
// The "redundant" work across 64 SIMD threads is free on GPU hardware.

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
