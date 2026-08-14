// GPU-side dequantization of Q1_0 GGUF blocks.
// Replaces the CPU wireQ10 + xfQ2/xfSign + f16f32 transforms.
//
// Each 18-byte Q1_0 block: [f16 scale (2 bytes)][16 sign bytes]
//
// mode 0 (codes): each sign byte → 2 code bytes via tgt2 LUT → 32 bytes out
// mode 1 (signs): each sign byte → 1 sign byte via signTable LUT → 16 bytes out
//
// Both modes also extract the f16 scale → f32 and write to the scales buffer.
// One thread per block. Output ranges never overlap between threads.

struct Params {
  numBlocks: u32,
  mode: u32,   // 0 = codes (xfQ2), 1 = signs (xfSign)
  _p0: u32,
  _p1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> raw: array<u32>;          // raw Q1_0 data
@group(0) @binding(2) var<storage, read> tgt2: array<u32>;         // 128 u32s (512 bytes)
@group(0) @binding(3) var<storage, read> signTable: array<u32>;    // 64 u32s (256 bytes)
@group(0) @binding(4) var<storage, read_write> outBuf: array<u32>; // codes or signs
@group(0) @binding(5) var<storage, read_write> scales: array<f32>;

// Read a byte from a byte-addressed u32 array
fn readByte(arr: ptr<storage, array<u32>>, byteOff: u32) -> u32 {
  return (arr[byteOff >> 2u] >> (8u * (byteOff & 3u))) & 0xFFu;
}

// Convert f16 bits → f32 bits (IEEE 754)
fn f16ToBits(h: u32) -> u32 {
  let sign = select(0u, 0x80000000u, (h & 0x8000u) != 0u);
  let exp = (h >> 10u) & 31u;
  let mant = h & 1023u;
  if (exp == 0u) {
    return sign | (mant << 13u);
  } else if (exp == 31u) {
    return sign | 0x7F800000u | (mant << 13u);
  }
  return sign | ((exp + 112u) << 23u) | (mant << 13u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockIdx = gid.x;
  if (blockIdx >= p.numBlocks) { return; }

  let baseByte = blockIdx * 18u;

  // Read f16 scale (2 bytes at baseByte, little-endian)
  let scaleLo = readByte(&raw, baseByte);
  let scaleHi = readByte(&raw, baseByte + 1u);
  let f16Bits = scaleLo | (scaleHi << 8u);
  scales[blockIdx] = bitcast<f32>(f16ToBits(f16Bits));

  if (p.mode == 0u) {
    // Codes mode: 16 sign bytes → 32 code bytes → 8 u32 words
    var words: array<u32, 8>;
    for (var w = 0u; w < 8u; w = w + 1u) { words[w] = 0u; }

    for (var j = 0u; j < 16u; j = j + 1u) {
      let sb = readByte(&raw, baseByte + 2u + j);

      // tgt2 lookup: 256 entries × 2 bytes each = 512 bytes
      // Entry sb is at byte offset 2*sb. Since 2*sb is always even,
      // both code bytes fall within the same u32 word.
      let lutByteOff = 2u * sb;
      let lutWord = tgt2[lutByteOff >> 2u];
      let lutShift = 8u * (lutByteOff & 3u);
      let code0 = (lutWord >> lutShift) & 0xFFu;
      let code1 = (lutWord >> (lutShift + 8u)) & 0xFFu;

      // Pack 2 code bytes into the output words
      let outByteOff = 2u * j;  // within this block's 32-byte output
      let wordIdx = outByteOff >> 2u;
      let byteIdx = outByteOff & 3u;
      words[wordIdx] = words[wordIdx] | (code0 << (8u * byteIdx)) | (code1 << (8u * (byteIdx + 1u)));
    }

    for (var w = 0u; w < 8u; w = w + 1u) {
      outBuf[blockIdx * 8u + w] = words[w];
    }
  } else {
    // Signs mode: 16 sign bytes → 16 sign bytes → 4 u32 words
    var words: array<u32, 4>;
    for (var w = 0u; w < 4u; w = w + 1u) { words[w] = 0u; }

    for (var j = 0u; j < 16u; j = j + 1u) {
      let sb = readByte(&raw, baseByte + 2u + j);

      // signTable lookup: 1 byte per entry, 4 entries per u32
      let lutWord = signTable[sb >> 2u];
      let signByte = (lutWord >> (8u * (sb & 3u))) & 0xFFu;

      let wordIdx = j >> 2u;
      let byteIdx = j & 3u;
      words[wordIdx] = words[wordIdx] | (signByte << (8u * byteIdx));
    }

    for (var w = 0u; w < 4u; w = w + 1u) {
      outBuf[blockIdx * 4u + w] = words[w];
    }
  }
}
