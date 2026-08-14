// F32 matmul for vision tower: C[M,N] = A[M,K] × W[N,K] + bias[N]
// W is stored as [N, K] (row-major, output × input) in F32.
// One thread per output column; loops over K (no subgroups — avoids SG mismatch).
//
// Supports split outputs: when N0+N1+N2 = N, routes output columns to
// out0/out1/out2 by range (for fused QKV projection). When N1=N2=0,
// everything goes to out0 (single-output mode).
struct Params {
  M: u32,        // number of input rows (seq_len / num_patches)
  N: u32,        // total output features
  K: u32,        // input features
  N0: u32,       // first split (Q or full N)
  N1: u32,       // second split (K, or 0 for single output)
  N2: u32,       // third split (V, or 0 for single output)
  hasBias: u32,  // 1 = add bias, 0 = no bias
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [M, K]
@group(0) @binding(2) var<storage, read> w: array<f32>;        // [N, K]
@group(0) @binding(3) var<storage, read> bias: array<f32>;     // [N] (or dummy)
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;  // [M, N0]
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;  // [M, N1] (unused if N1=0)
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;  // [M, N2] (unused if N2=0)

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let n = wg.x * 64u + lid.x;  // one thread per output column
  if (n >= p.N) { return; }

  let w_base = n * p.K;
  let b = select(0.0, bias[n], p.hasBias != 0u);

  for (var m = 0u; m < p.M; m++) {
    let x_base = m * p.K;
    var acc = 0.0;
    for (var k = 0u; k < p.K; k = k + 1u) {
      acc = acc + x[x_base + k] * w[w_base + k];
    }
    let val = acc + b;
    if (n < p.N0) {
      out0[m * p.N0 + n] = val;
    } else if (n < p.N0 + p.N1) {
      out1[m * p.N1 + (n - p.N0)] = val;
    } else {
      out2[m * p.N2 + (n - p.N0 - p.N1)] = val;
    }
  }
}
