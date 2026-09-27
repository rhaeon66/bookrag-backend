# Deploy: Vercel (frontend) + your PC via Cloudflare Tunnel (backend)

The backend runs on your own machine; Cloudflare Tunnel gives it a public HTTPS URL
without opening any router ports. It is only online while your PC is on and the tunnel runs.

## 1. LLM
Either keep local Ollama (`ollama serve`, `ollama pull qwen2.5:3b`; nothing to change),
or use Ollama Cloud: in `backend/.env` set
```
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<key from https://ollama.com/settings/keys>
LLM_MODEL=gpt-oss:20b
```
Never commit `.env` (it is gitignored).

## 2. Run the backend
```
cd backend
make -C .. ingest                       # once, if data/chroma is missing
CORS_ORIGINS=https://<your-app>.vercel.app ./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Check http://localhost:8000/api/health.

## 3. Install cloudflared (Debian/Ubuntu)
```
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
```

## 4a. Quick tunnel (no account, no domain; URL changes every restart)
```
cloudflared tunnel --url http://localhost:8000
```
It prints `https://<random>.trycloudflare.com`. Use that as the backend URL.
Every restart gives a new URL, so you must update Vercel's env var and redeploy each time.

## 4b. Named tunnel (stable URL; needs a domain added to a free Cloudflare account)
```
cloudflared tunnel login
cloudflared tunnel create bookrag
cloudflared tunnel route dns bookrag api.<yourdomain.com>
```
Create `~/.cloudflared/config.yml`:
```
tunnel: bookrag
credentials-file: /home/<you>/.cloudflared/<TUNNEL-UUID>.json
ingress:
  - hostname: api.<yourdomain.com>
    service: http://localhost:8000
  - service: http_status:404
```
Run it: `cloudflared tunnel run bookrag`
Start on boot: `sudo cloudflared service install` (copy config to /etc/cloudflared/ first).
A domain costs money; without one, use 4a, or ngrok's free static domain as an alternative.

## 5. Vercel
1. Import the `bookrag-frontend` repo (Next.js auto-detected).
2. Env var `NEXT_PUBLIC_API_BASE_URL=https://<tunnel-url>` (no trailing slash), Production + Preview.
3. Deploy. Changing the env var later requires a redeploy.
4. Restart the backend with `CORS_ORIGINS=https://<your-app>.vercel.app` (comma-separate extra origins).

## Notes
- The tunnel URL is public: anyone with it can query your backend and use your LLM quota.
- `Dockerfile` is kept for hosts that run containers; it is not needed for this setup.
