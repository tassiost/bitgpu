// Initialize tiled attention state: m=-inf, l=0, acc=0.
struct Params { SH: u32, D: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> l_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc_out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = wg.x * 64u + lid.x;
  if (tid >= p.SH * p.D) { return; }
  let sh = tid / p.D;
  let d = tid % p.D;
  if (d == 0u) { m_out[sh] = -1e30; l_out[sh] = 0.0; }
  acc_out[tid] = 0.0;
}
