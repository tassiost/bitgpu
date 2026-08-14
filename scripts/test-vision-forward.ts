// Test the vision tower forward pass with the real mmproj file.
// This verifies:
// 1. GGUF header parsing
// 2. Q8_0 dequantization
// 3. Weight loading
// 4. Vision forward pass (CPU reference implementation)
//
// Run: npx tsx test-vision-forward.ts

import { loadVisionWeights, visionForwardCpu, preprocessImage, type ImageInput } from '../src/vision'

const MMPROJ_URL = 'file:///Users/tassio/Downloads/Bonsai-27B-mmproj-Q8_0.gguf'
const fs = await import('fs')

// Custom fetchRange for local file
async function fetchRange(url: string, off: number, len: number): Promise<ArrayBuffer> {
  const path = url.replace('file://', '')
  const fd = fs.openSync(path, 'r')
  const buf = Buffer.alloc(len)
  fs.readSync(fd, buf, 0, len, off)
  fs.closeSync(fd)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
}

console.log('Loading vision weights from mmproj...')
const t0 = performance.now()
const { weights, config } = await loadVisionWeights(MMPROJ_URL, fetchRange, (loaded, total) => {
  if (loaded % 50 === 0 || loaded === total) {
    process.stdout.write(`\r  Loaded ${loaded}/${total} tensors...`)
  }
})
const loadMs = performance.now() - t0
console.log(`\n  Loaded ${config.depth} layers in ${loadMs.toFixed(0)}ms`)
console.log(`  Config: hidden=${config.hidden_size}, heads=${config.num_heads}, head_dim=${config.head_dim}, proj=${config.out_hidden_size}`)

// Create a small test image (32x32 RGB, all red)
const width = 32, height = 32
const rgb = new Float32Array(3 * width * height)
for (let i = 0; i < width * height; i++) {
  rgb[i * 3] = 1.0      // R
  rgb[i * 3 + 1] = 0.0  // G
  rgb[i * 3 + 2] = 0.0  // B
}

const image: ImageInput = { rgb, width, height, frames: 1 }

// Preprocess
const { patches, numPatches, gridThw } = preprocessImage(image, config)
console.log(`\nPreprocessed: ${numPatches} patches, gridThw=${gridThw}`)
console.log(`  Patch dim: ${patches.length / numPatches} (expected ${3 * 2 * 16 * 16})`)

// Run vision forward
console.log('\nRunning vision forward pass (CPU)...')
const t1 = performance.now()
const result = visionForwardCpu([image], weights, config)
const fwdMs = performance.now() - t1
console.log(`  Forward pass: ${fwdMs.toFixed(0)}ms`)
console.log(`  Image embeds: ${result.imageEmbeds.length} floats (${result.numPatches} patches × ${config.out_hidden_size} dims)`)
console.log(`  Expected: ${result.numPatches * config.out_hidden_size}`)
console.log(`  DeepStack features: ${result.deepstackFeatures.length}`)

// Verify output
const embeds = result.imageEmbeds
let minVal = Infinity, maxVal = -Infinity, sum = 0
for (let i = 0; i < embeds.length; i++) {
  if (embeds[i] < minVal) minVal = embeds[i]
  if (embeds[i] > maxVal) maxVal = embeds[i]
  sum += embeds[i]
}
console.log(`\nOutput stats: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}, mean=${(sum / embeds.length).toFixed(4)}`)
console.log(`  First 5 values: [${embeds.slice(0, 5).map(v => v.toFixed(4)).join(', ')}]`)

// Check for NaN/Inf
let nanCount = 0, infCount = 0
for (let i = 0; i < embeds.length; i++) {
  if (Number.isNaN(embeds[i])) nanCount++
  if (!Number.isFinite(embeds[i])) infCount++
}
console.log(`  NaN: ${nanCount}, Inf: ${infCount}`)

if (nanCount > 0 || infCount > 0) {
  console.log('\n❌ FAIL: output contains NaN or Inf')
  process.exit(1)
}
if (minVal === maxVal) {
  console.log('\n❌ FAIL: output is constant (all same value)')
  process.exit(1)
}

console.log('\n✅ Vision forward pass works! Output is valid (non-constant, no NaN/Inf)')
console.log(`   ${result.numPatches} image tokens × ${config.out_hidden_size} dims = ${result.imageEmbeds.length * 4} bytes`)
