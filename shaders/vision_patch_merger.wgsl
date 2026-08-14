// Vision patch merger: pixel shuffle (2x2 spatial merge).
// Input: [num_patches, 1152] arranged as [t, h_blk, w_blk, h_intra, w_intra, 1152]
//   (consecutive groups of 4 patches form one 2x2 merge block)
// Output: [num_merged, 4608] where num_merged = t * (h/2) * (w/2)
// Each merged patch concatenates 4 consecutive patches from the input.
//
// The linear layers (mm.0: 4608→4608, mm.2: 4608→5120) use vision_matmul.
// GELU between them uses vision_gelu.
struct Params {
  t: u32,           // temporal patches
  h: u32,           // height patches (before merge)
  w: u32,           // width patches (before merge)
  hidden: u32,      // 1152
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [t*h*w, hidden]
@group(0) @binding(2) var<storage, read_write> out: array<f32>;  // [t*(h/2)*(w/2), hidden*4]

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let merged_h = p.h / 2u;
  let merged_w = p.w / 2u;
  let num_merged = p.t * merged_h * merged_w;
  let merged_dim = p.hidden * 4u;  // 4608

  // Each workgroup handles one merged patch, lanes split the 4608-dim output
  let m = wid.x;
  if (m >= num_merged) { return; }

  // Patches are in [t, h_blk, w_blk, h_intra, w_intra] order.
  // Consecutive groups of 4 patches form one merge block.
  // Slot order: (ih=0,iw=0), (ih=0,iw=1), (ih=1,iw=0), (ih=1,iw=1) = slot 0,1,2,3
  for (var i = lid.x; i < merged_dim; i = i + 64u) {
    let slot = i / p.hidden;     // 0..3
    let within = i % p.hidden;   // 0..1151
    let orig_idx = m * 4u + slot;  // consecutive in [h_blk, w_blk, h_intra, w_intra] order
    out[m * merged_dim + i] = x[orig_idx * p.hidden + within];
  }
}
