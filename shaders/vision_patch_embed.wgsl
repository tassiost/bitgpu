// Vision patch embedding: Conv3d with stride == kernel (pointwise per patch).
// Each patch is 3 * temporal_patch_size * patch_size * patch_size = 1536 values → 1152 outputs.
// This is a single matmul (1536 → 1152) with bias, applied per patch.
// One workgroup per patch, subgroup-parallel over the output dimension.
struct Params {
  num_patches: u32,   // number of patches (batch * temporal * spatial)
  in_dim: u32,        // 1536 = 3 * 2 * 16 * 16
  out_dim: u32,       // 1152
  _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> patches: array<f32>;    // [num_patches, in_dim]
@group(0) @binding(2) var<storage, read> weight: array<f32>;     // [out_dim, in_dim]
@group(0) @binding(3) var<storage, read> bias: array<f32>;       // [out_dim]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;  // [num_patches, out_dim]

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  // Each workgroup computes one output column across all patches
  let out_col = wg.x * 64u + lid.x;
  if (out_col >= p.out_dim) { return; }

  for (var patch_idx = 0u; patch_idx < p.num_patches; patch_idx++) {
    let patch_base = patch_idx * p.in_dim;
    let w_base = out_col * p.in_dim;
    var sum = bias[out_col];
    for (var i = 0u; i < p.in_dim; i++) {
      sum = sum + patches[patch_base + i] * weight[w_base + i];
    }
    out[patch_idx * p.out_dim + out_col] = sum;
  }
}
