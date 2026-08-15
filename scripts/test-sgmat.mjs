// Test SubgroupMatrix feature
import { createServer } from 'node:http'
import { join, dirname, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFileSync } from 'node:fs'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' }

const server = createServer((req, res) => {
  let path = req.url.split('?')[0]
  if (path === '/') path = '/examples/test-sgmat-basic.html'
  const filePath = join(root, path)
  try {
    const data = readFileSync(filePath)
    res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' })
    res.end(data)
  } catch {
    res.writeHead(404); res.end('Not found')
  }
})

server.listen(8890, async () => {
  const { chromium } = await import('playwright-core')
  const fs = await import('node:fs')
  const exe = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  if (!fs.existsSync(exe)) { console.error('Chrome not found'); process.exit(1) }

  const browser = await chromium.launch({
    headless: false,
    executablePath: exe,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--enable-webgpu-developer-features'],
  })
  const page = await browser.newPage()
  page.on('console', msg => console.log('  [console]', msg.text()))
  page.on('pageerror', err => console.log('  [pageerror]', err.message))
  await page.goto('http://localhost:8890/examples/test-sgmat-basic.html', { waitUntil: 'networkidle0' })
  await new Promise(r => setTimeout(r, 5000))
  const text = await page.evaluate(() => document.getElementById('out').textContent)
  console.log(text)
  await browser.close()
  server.close()
})
