# Deployment Guide - Fly.io (Separate Apps)

## Architecture

Deploy three separate Fly.io apps:

1. **modern-wisdom-qdrant** - Vector database (Qdrant)
2. **modern-wisdom-phoenix** - Observability dashboard (optional)
3. **modern-wisdom-rag** - Main application (Chainlit + FastAPI)

Apps communicate via Fly's internal networking: `http://<app-name>.internal:<port>`

## Prerequisites

- Fly.io CLI installed: `curl -L https://fly.io/install.sh | sh` and then run

    ```bash
    echo 'export FLYCTL_INSTALL="/Users/ettyekhon/.fly"' >> ~/.zshrc
    echo 'export PATH="$FLYCTL_INSTALL/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    ```

- Fly.io account: `fly auth signup` or `fly auth login`
- OpenAI API key

## Step-by-Step Deployment

### 1. Deploy Qdrant

**Note:** Fly.io volumes are pinned to specific physical hosts. For production, create multiple volumes for redundancy. For development/testing, a single volume is acceptable.

```bash
# Create the app first (required before creating volumes)
fly apps create modern-wisdom-qdrant

# Create volume(s) for Qdrant data (persistent storage)
# You may see a message to create two or more volumes per application to avoid downtime. For development: single volume is fine
fly volumes create qdrant_data --app modern-wisdom-qdrant --size 10 --region iad

# Deploy Qdrant
fly deploy --config fly.qdrant.toml

# Verify Qdrant is running
fly status --app modern-wisdom-qdrant
fly ips list --app modern-wisdom-qdrant

curl https://modern-wisdom-qdrant.fly.dev/healthz

# Access Qdrant dashboard
open https://modern-wisdom-qdrant.fly.dev/dashboard#/collections

```

**Volume Warning:** Fly.io volumes are pinned to specific hosts. If that host fails, the volume becomes unavailable. For production, consider:

- Creating 2+ volumes and using Qdrant's replication features
- Using Qdrant Cloud (managed service) instead
- Regular backups of the volume data

### 2. Deploy Phoenix (Optional)

**Note:** Phoenix data is less critical than Qdrant. A single volume is usually sufficient, but you can create multiple for redundancy.

```bash
# Create the app first (required before creating volumes)
fly apps create modern-wisdom-phoenix

# Create volume for Phoenix data
fly volumes create phoenix_data --app modern-wisdom-phoenix --size 3 --region iad

# Deploy Phoenix
fly deploy --config fly.phoenix.toml

# Verify Phoenix is running
fly status --app modern-wisdom-phoenix
fly ips list --app modern-wisdom-phoenix
# Public URL uses standard HTTPS (no port needed)
curl https://modern-wisdom-phoenix.fly.dev

# Access Phoenix dashboard
https://modern-wisdom-phoenix.fly.dev/projects
```

### 3. Deploy Main Application

```bash
# Create the app first
fly apps create modern-wisdom-rag

# Set required secrets
fly secrets set OPENAI_API_KEY=sk-your-key-here --app modern-wisdom-rag

# Deploy main app
fly deploy --config fly.toml

# Verify app is running
fly status --app modern-wisdom-rag
curl https://modern-wisdom-rag.fly.dev/healthz
```

**Note:** If you get a DNS error (`DNS_PROBE_FINISHED_NXDOMAIN`), the app may not have IP addresses allocated. Check:

```bash
# Check if app exists and has IPs
fly status --app modern-wisdom-rag
fly ips list --app modern-wisdom-rag
```

### 4. Upsert Embeddings

After deployment, upsert embeddings so episodes are searchable:

```bash
# Option A: Run locally, pointing to Fly Qdrant
export QDRANT_URL="https://modern-wisdom-qdrant.fly.dev"
export QDRANT_API_KEY=your-actual-api-key
uv run mw-rag upsert-batch \
  --episode-list data/tmp/epids_2018_2025.txt \
  --emb-v "BAAI/bge-small-en-v1.5" \
  --set-live

# Option B: Run via Fly SSH (if you've copied embeddings to the app)
fly ssh console --app modern-wisdom-rag
# Then inside the console:
uv run mw-rag upsert-batch \
  --episode-list /app/data/tmp/epids_2018_2025.txt \
  --emb-v "BAAI/bge-small-en-v1.5" \
  --set-live
```

**Note:** Initial upsert takes ~30+ minutes for ~1000 episodes.

### 5. Verify Everything Works

```bash
# Check Qdrant collection (public HTTPS, no port needed)
export QDRANT_URL="https://modern-wisdom-qdrant.fly.dev"
uv run mw-rag check --emb-v "BAAI/bge-small-en-v1.5"

# Expected output
# [10:56:55] INFO     Connecting to Fly.io Qdrant:                          qdrant_ops.py:56
#                     host=modern-wisdom-qdrant.fly.dev, port=443,                          
#                     https=True, http2=True, timeout=120s                                  
#            INFO     HTTP Request: GET https://modern-wisdom-qdrant.fly.dev _client.py:1025
#                     "HTTP/1.1 200 OK"                                                     
#            INFO     HTTP Request: GET                                      _client.py:1025
#                     https://modern-wisdom-qdrant.fly.dev/collections/mw_ch                
#                     unks_baai_bge_small_en_v1.5 "HTTP/2 200 OK"                           
# [10:56:56] INFO     HTTP Request: POST                                     _client.py:1025
#                     https://modern-wisdom-qdrant.fly.dev/collections/mw_ch                
#                     unks_baai_bge_small_en_v1.5/points/count "HTTP/2 200                  
#                     OK"                                                                   
# mw_chunks_baai_bge_small_en_v1.5: status=green, vector_size=384, distance=Cosine, 
# points=40001
#            INFO     HTTP Request: GET                                      _client.py:1025
#                     https://modern-wisdom-qdrant.fly.dev/collections/mw_ch                
#                     unks_baai_bge_small_en_v1.5/aliases "HTTP/2 200 OK"                   
# Aliases for mw_chunks_baai_bge_small_en_v1.5: ['mw_chunks_live']


# Test main app
curl https://modern-wisdom-rag.fly.dev/info

# Access Chainlit UI
open https://modern-wisdom-rag.fly.dev

# Access Phoenix dashboard (if deployed)
open https://modern-wisdom-phoenix.fly.dev

# Access Qdrant dashboard (view collections and points)
open https://modern-wisdom-qdrant.fly.dev/dashboard#/collections
```

## Configuration Files

- `fly.toml` - Main application
- `fly.qdrant.toml` - Qdrant service
- `fly.phoenix.toml` - Phoenix service (optional)

## Internal Networking

Fly apps communicate via internal DNS:

- Qdrant (internal): `http://modern-wisdom-qdrant.internal:6333`
- Phoenix (internal): `http://modern-wisdom-phoenix.internal:6006`

Public access (via Fly proxy) uses HTTPS without ports:

- Qdrant (public): `https://modern-wisdom-qdrant.fly.dev`
- Qdrant Dashboard: `https://modern-wisdom-qdrant.fly.dev/dashboard#/collections`
- Phoenix (public): `https://modern-wisdom-phoenix.fly.dev`

These URLs are already configured in `fly.toml` environment variables.

## Episodes Availability

**Episodes will only be searchable if:**

1. Qdrant collection exists with alias `mw_chunks_live`
2. Embeddings have been upserted (see step 4)
3. `INDEX_VERSION` environment variable matches the collection alias

**To check:**

```bash
export QDRANT_URL="https://modern-wisdom-qdrant.fly.dev"
uv run mw-rag check --emb-v "BAAI/bge-small-en-v1.5"
```

## Troubleshooting

### Check app status

```bash
fly status --app modern-wisdom-rag
fly status --app modern-wisdom-qdrant
fly status --app modern-wisdom-phoenix
```

### View logs

```bash
fly logs --app modern-wisdom-rag -n
fly logs --app modern-wisdom-qdrant -n
fly logs --app modern-wisdom-phoenix -n
```

### Troubleshoot Phoenix Dashboard

If Phoenix dashboard isn't accessible:

```bash
# Check Phoenix logs for errors (recent logs, non-streaming)
fly logs --app modern-wisdom-phoenix -n

# Test if Phoenix is responding
curl -v https://modern-wisdom-phoenix.fly.dev

# Check if the machine is running
fly status --app modern-wisdom-phoenix

# If you see "Out of memory" (OOM) kills in logs, scale up memory:
fly scale vm shared-cpu-1x --memory 1024 --app modern-wisdom-phoenix

# Or redeploy with updated config (includes 1GB memory):
fly deploy --config fly.phoenix.toml

# Restart Phoenix if needed
fly apps restart modern-wisdom-phoenix
```

#### Common Issue: Out of Memory (OOM) Kills

If Phoenix or the main RAG app is being killed repeatedly with "Out of memory" errors, they need more RAM. The default 256MB is insufficient.

**For Phoenix:** Update `fly.phoenix.toml` to include:

```toml
[vm]
  memory_mb = 1024
  cpu_kind = "shared"
  cpus = 1
```

Then redeploy, or scale the existing machine:

```bash
fly scale vm shared-cpu-1x --memory 1024 --app modern-wisdom-phoenix
```

**For Main RAG App:** Update `fly.toml` to include the same `[vm]` section (already included in the default config). If you see OOM kills, scale the existing machine:

```bash
fly scale vm shared-cpu-1x --memory 1024 --app modern-wisdom-rag
```

### SSH into app

```bash
fly ssh console --app modern-wisdom-rag
```

### Update secrets

```bash
fly secrets set OPENAI_API_KEY=sk-new-key --app modern-wisdom-rag
```

## Scaling

To scale apps:

```bash
fly scale count 2 --app modern-wisdom-rag  # Scale main app
fly scale vm shared-cpu-1x --app modern-wisdom-qdrant  # Upgrade Qdrant VM
```

## Cost Considerations

- Qdrant volume: ~$0.15/GB/month (per volume)
- Phoenix volume: ~$0.15/GB/month (if used)
- App compute: Based on VM size and usage
- Network: Internal networking is free

**Volume Redundancy:** For production, plan for 2+ volumes per service, doubling storage costs but providing redundancy.

## Alternative: Single App with Multiple Services

If you prefer a single app, you can use Fly machines to run multiple services, but separate apps are recommended for better isolation and scaling.
