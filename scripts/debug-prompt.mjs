// Quick debug driver to inspect the prompt token sequence
import { createReadStream, existsSync, statSync, unlinkSync } from 'node:fs'
import { join, dirname, normalize, sep, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { createServer } from 'node:http'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const downloads = join(homedir(), 'Downloads')
const MIME = { '.html':'text/html', '.js':'text/javascript', '.json':'application/json', '.gguf':'application/octet-stream', '.bin':'application/octet-stream', '.map':'application/json', '.d.ts':'text/plain', '.css':'text/css', '.wgsl':'text/plain' }

function resolvePath(rel) {
  const r = join(root, rel.replace(/^\//, '')); if (existsSync(r)) return r
  const d = join(downloads, rel.replace(/^\//, '')); if (existsSync(d)) return d
  return null
}

const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const rel = decodeURIComponent(normalize(url.pathname).replace(/^([/\\])+/, ''))
  if (rel.split(sep).includes('..')) { res.writeHead(403).end(); return }
  const path = resolvePath(rel)
  if (!path || !existsSync(path) || !statSync(path).isFile()) { res.writeHead(404).end(); return }
  const range = req.headers.range
  if (range) {
    const m = /bytes=(\d+)-(\d*)/.exec(range)
    if (m) {
      const sz = statSync(path).size
      const start = parseInt(m[1], 10), end = m[2] ? parseInt(m[2], 10) : sz - 1
      if (start >= sz || end >= sz || start > end) { res.writeHead(416).end(); return }
      res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${sz}`, 'Accept-Ranges': 'bytes', 'Content-Length': String(end-start+1), 'Content-Type': MIME[extname(path).toLowerCase()] || 'application/octet-stream' })
      createReadStream(path, { start, end }).pipe(res); return
    }
  }
  res.writeHead(200, { 'Content-Type': MIME[extname(path).toLowerCase()] || 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Content-Length': String(statSync(path).size) })
  createReadStream(path).pipe(res)
})

server.listen(8888, async () => {
  const { chromium } = await import('playwright-core')
  const exe = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  const browser = await chromium.launch({ headless: false, executablePath: exe, args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox'] })
  const page = await browser.newPage()
  page.on('console', msg => console.log('[console]', msg.text()))
  await page.goto('http://localhost:8888/examples/debug-prompt.html', { waitUntil: 'domcontentloaded' })

  // Wait for completion
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 2000))
    const content = await page.evaluate(() => document.getElementById('out')?.textContent || '')
    if (content.includes('TEST_DONE')) break
  }
  const output = await page.evaluate(() => document.getElementById('out')?.textContent || '')
  console.log(output)
  await browser.close()
  server.close()
  try { unlinkSync(join(root, 'examples', 'debug-prompt.html')) } catch {}
})
