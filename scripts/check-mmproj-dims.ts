import { loadVisionWeights } from '../src/vision'
import * as fs from 'fs'

async function fetchRange(url: string, off: number, len: number): Promise<ArrayBuffer> {
  const path = url.replace('file://', '')
  const fd = fs.openSync(path, 'r')
  const buf = Buffer.alloc(len)
  fs.readSync(fd, buf, 0, len, off)
  fs.closeSync(fd)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
}

const { weights, config } = await loadVisionWeights('file:///Users/tassio/Downloads/Bonsai-27B-mmproj-Q8_0.gguf', fetchRange)
console.log('out_hidden_size:', config.out_hidden_size)
console.log('mm2 weight length:', weights.mergerMm2Weight.length)
console.log('expected [5120, 4608]:', 5120 * 4608)
console.log('expected [3584, 4608]:', 3584 * 4608)
console.log('mm2 bias length:', weights.mergerMm2Bias.length)
console.log('mm0 weight length:', weights.mergerMm0Weight.length)
console.log('expected [4608, 4608]:', 4608 * 4608)
console.log('patchEmbdWeight length:', weights.patchEmbdWeight.length)
console.log('expected [1152, 1536]:', 1152 * 1536)
console.log('patchEmbdBias length:', weights.patchEmbdBias.length)
console.log('expected [1152]:', 1152)
console.log('positionEmbd length:', weights.positionEmbd.length)
console.log('expected [2304, 1152]:', 2304 * 1152)
