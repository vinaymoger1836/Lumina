# Deploying Lumina (API + n8n) to AWS EC2 — Phase 3

This runbook stands up the Lumina FastAPI backend and n8n on a **single free-tier
EC2 instance** with Docker Compose. n8n calls the API over the internal Docker
network; you reach the n8n UI over an SSH tunnel (so Google OAuth works without a
domain or HTTPS cert).

```
┌──────────── EC2 t3.micro (Ubuntu, 1 GB RAM + 2 GB swap) ────────────┐
│  docker compose:                                                     │
│    lumina-api  (FastAPI :8000, internal only)                        │
│    n8n         (:5678, reached via SSH tunnel)                        │
│  n8n → http://lumina-api:8000/webhooks/*  (internal Docker DNS)       │
└──────────────────────────────────────────────────────────────────────┘
```

> **Free-tier reality check:** the t3.micro free allowance is **12 months**, then
> ~$7–8/mo. Set a **billing alarm** (Billing → Budgets) now. 1 GB RAM is tight for
> torch + sentence-transformers + n8n — the **2 GB swap file** below is what keeps
> it from OOM-ing.

---

## 1. Launch the EC2 instance

AWS Console → EC2 → **Launch instance**:

- **Name:** `lumina`
- **AMI:** Ubuntu Server 24.04 LTS (free-tier eligible)
- **Instance type:** `t3.micro` (or `t2.micro`) — free-tier eligible
- **Key pair:** create/download one (e.g. `lumina.pem`) — you SSH with this
- **Storage:** bump the root volume to **20 GB** gp3 (the default 8 GB is too small
  once Docker images + the torch layers land; 30 GB is still within free tier)
- **Security group** — inbound rules:
  | Type | Port | Source | Why |
  |------|------|--------|-----|
  | SSH  | 22   | **My IP** | admin + the tunnel |
  | *(nothing else)* | | | API and n8n stay private; you tunnel in |

Launch, then **Elastic IP** → Allocate → Associate with this instance (gives a
stable public IP that survives reboots; free while attached to a running box).

---

## 2. First-boot setup (SSH in)

```bash
ssh -i lumina.pem ubuntu@<ELASTIC_IP>

# 2 GB swap — the safety net for torch memory on a 1 GB box
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Docker + compose plugin
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu
newgrp docker   # apply group without re-login
```

---

## 3. Get the code + secrets onto the box

```bash
git clone https://github.com/vinaymoger1836/Lumina.git
cd Lumina
```

Create the `.env` on the server (it is **not** in git). Copy `.env.example` and
fill every value — use the **same** `GROQ/QDRANT/TAVILY` keys as local, the
`N8N_WEBHOOK_TOKEN` you generated, and `DIGEST_RECIPIENT`:

```bash
cp .env.example .env
nano .env     # paste real values
```

> Keep `N8N_HOST=localhost` / `WEBHOOK_URL=http://localhost:5678/` in `.env` — you
> access n8n through the tunnel, and this lets Google OAuth use a localhost redirect.

---

## 4. Build and start

```bash
docker compose up -d --build      # first build is slow (torch layer)
docker compose logs -f lumina-api # watch until "Application startup complete"
```

Sanity-check the API from the box:

```bash
curl localhost:8000/health        # ...but 8000 isn't published; instead:
docker compose exec lumina-api curl -s localhost:8000/health   # -> {"status":"ok"}
```

If the API OOM-restarts (check `docker compose logs`), the swap file usually
saves it; if not, you'll need a bigger instance (t3.small, ~$15/mo).

---

## 5. Reach the n8n UI (SSH tunnel)

From **your laptop** (not the server):

```bash
ssh -i lumina.pem -L 5678:localhost:5678 ubuntu@<ELASTIC_IP>
```

Leave that open, then browse to **http://localhost:5678** locally. Set up the
n8n owner account on first visit.

---

## 6. Google Cloud OAuth (Drive + Gmail)

One Google Cloud project covers both. Console → https://console.cloud.google.com

1. **Create project** `lumina`.
2. **APIs & Services → Enable APIs** → enable **Google Drive API** and **Gmail API**.
3. **OAuth consent screen** → External → fill app name/email → add **your Gmail as a
   Test user** (keep it in "Testing" mode; no verification needed for personal use).
4. **Credentials → Create credentials → OAuth client ID → Web application**.
   - You'll add the redirect URI n8n shows you in the next step.
5. In n8n (via the tunnel): **Credentials → New**:
   - **Google Drive OAuth2 API** → n8n shows a redirect URL like
     `http://localhost:5678/rest/oauth2-credential/callback` → paste that into the
     Google client's **Authorized redirect URIs** → paste the Google **Client ID +
     Secret** back into n8n → **Connect / Sign in with Google**.
   - Repeat for **Gmail OAuth2**.

> Because you're on `localhost` (via the tunnel), Google accepts the HTTP redirect.
> No domain or TLS cert required.

---

## 7. Import and wire the workflows

In n8n → **Workflows → Import from File**, import both:

- `n8n/workflows/drive_auto_ingest.json`
- `n8n/workflows/daily_digest_email.json`

Then in each, replace the placeholder nodes:

- **Drive trigger + Download** → select your **Google Drive OAuth2** credential;
  set **folderToWatch** to the Drive folder you'll drop PDFs into.
- **Gmail node** → select your **Gmail OAuth2** credential.
- The HTTP nodes already read `LUMINA_API_BASE`, `N8N_WEBHOOK_TOKEN`, and
  `DIGEST_RECIPIENT` from the container env — nothing to edit.

**Test:** open each workflow → **Execute Workflow** once.
- Drive: drop a PDF in the folder (or run manually against an existing file) →
  should POST to `/ingest/pdf` and 200.
- Digest: should hit `/webhooks/digest`, get the summary JSON, and send the email.

When both pass, toggle each workflow **Active**.

---

## 8. Verify end-to-end

- Drop a new PDF in the watched Drive folder → within ~1 min it's ingested
  (check API logs: `docker compose logs -f lumina-api`).
- Wait for 08:00 (instance timezone = `GENERIC_TIMEZONE`) or Execute the digest
  workflow manually → digest email arrives at `DIGEST_RECIPIENT`.

---

## Operations

| Task | Command (on the box) |
|------|----------------------|
| Restart everything | `docker compose restart` |
| Update after a `git pull` | `git pull && docker compose up -d --build` |
| Tail API logs | `docker compose logs -f lumina-api` |
| Check memory/swap | `free -h` |
| Stop (save nothing extra) | `docker compose down` (keeps volumes) |

**Costs to watch:** stop or terminate the instance when done experimenting; a
running t3.micro past the 12-month free window bills hourly. Terminating deletes
the n8n volume (workflows + Google creds) — re-import if you rebuild.
