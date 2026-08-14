// Vision GELU MLP: 1152 → 4304 → 1152 (GELU activation, pytorch_tanh variant).
// Two matmuls: gate (1152→4304) + down (4304→1152), with GELU in between.
// GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// This shader does the GELU activation only (elementwise). The matmuls use
// the existing matmul_split_sg shader with fp16 weights (HQQ 4-bit dequant).
const GELU_COEF: f32 = 0.7978845608028654; // sqrt(2/pi)

struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  let v = x[i];
  let inner = GELU_COEF * (v + 0.044715 * v * v * v);
  // Clamp to avoid tanh NaN: WGSL tanh uses (e^x - e^(-x))/(e^x + e^(-x))
  // which overflows to Inf/Inf=NaN for |inner| > ~40. tanh(15) ≈ 1.0 to f32 precision.
  let clamped = clamp(inner, -15.0, 15.0);
  y[i] = 0.5 * v * (1.0 + tanh(clamped));
}
