// Finalize tiled attention: divide acc by l.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, D: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> l_in: array<f32>;
@group(0) @binding(2) var<storage, read> acc_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let l = l_in[idx];
  let W4 = p.D / 4u;
  for (var w = lane; w < W4; w = w + SG) {
    let ab = idx * p.D + w * 4u;
    out[ab] = acc_in[ab] / l;
    out[ab + 1u] = acc_in[ab + 1u] / l;
    out[ab + 2u] = acc_in[ab + 2u] / l;
    out[ab + 3u] = acc_in[ab + 3u] / l;
  }
}
