// Inspect the mmproj GGUF file: list all tensor names, types, and dimensions.
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const downloads = join(homedir(), 'Downloads')

async function main() {
  const { parseGgufHeader } = await import(join(root, 'dist/gguf.js'))
  const url = `file://${join(downloads, 'Bonsai-27B-mmproj-Q8_0.gguf')}`

  // Fetch the first 4MB for the header
  const res = await fetch(`http://localhost:8888/Bonsai-27B-mmproj-Q8_0.gguf`, {
    headers: { Range: 'bytes=0-4194303' }
  }).catch(() => null)

  if (!res || !res.ok) {
    // Direct file read fallback
    const fs = await import('node:fs/promises')
    const buf = await fs.readFile(join(downloads, 'Bonsai-27B-mmproj-Q8_0.gguf'))
    const header = parseGgufHeader(buf.buffer)
    printTensors(header)
    return
  }

  const buf = await res.arrayBuffer()
  const header = parseGgufHeader(buf)
  printTensors(header)
}

function printTensors(header) {
  const { meta, tensors } = header
  console.log('=== Metadata ===')
  for (const [k, v] of Object.entries(meta)) {
    if (k.startsWith('clip.') || k.startsWith('mmproj.') || k.includes('vision') || k.includes('image')) {
      console.log(`  ${k} = ${v}`)
    }
  }
  console.log(`\n=== Tensors (${Object.keys(tensors).length} total) ===`)
  const names = Object.keys(tensors).sort()
  // Group by prefix
  const groups = {}
  for (const name of names) {
    const prefix = name.split('.').slice(0, 3).join('.')
    if (!groups[prefix]) groups[prefix] = []
    groups[prefix].push(name)
  }
  for (const [prefix, group] of Object.entries(groups)) {
    console.log(`\n  ${prefix}:`)
    for (const name of group) {
      const t = tensors[name]
      const dims = t.dims ? t.dims.join('×') : (t.ne ? t.ne.join('×') : '?')
      const type = t.type || t.dtype || '?'
      const size = t.size ? (t.size / 1e6).toFixed(2) + ' MB' : ''
      console.log(`    ${name} [${dims}] type=${type} ${size}`)
    }
  }
}

main().catch(e => { console.error(e); process.exit(1) })
