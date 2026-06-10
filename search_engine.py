"""
Codeforces Semantic Search Engine
Run: python search_engine.py --dataset CodeForces.csv
Then open: http://localhost:5000

AI Solutions powered by Groq (free tier, no credit card needed).
Semantic Search powered by Anthropic Voyage embeddings.

Set your API keys before running:
  Windows:
    set GROQ_API_KEY=your-key-here
    set ANTHROPIC_API_KEY=your-key-here
  Mac/Linux:
    export GROQ_API_KEY="your-key-here"
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os, json, pickle, re, csv, math, http.server, threading, urllib.parse, webbrowser, time
from pathlib import Path

# ── Optional numpy for faster cosine similarity ────────────────────────────
try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False

CACHE_PATH = "cf_embeddings.pkl"

# ── API KEYS ───────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── DATASET LOADING ────────────────────────────────────────────────────────
def load_dataset(path: str) -> list:
    problems = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_tags = row.get("tags", "") or ""
            tags = re.findall(r"'([^']+)'", raw_tags)

            rating_raw = row.get("rating", "") or ""
            try:
                rating = int(float(rating_raw))
            except (ValueError, TypeError):
                rating = 0

            contest_id = row.get("contestId", "") or ""
            index      = row.get("index", "") or ""
            url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}" if contest_id else ""

            problems.append({
                "id":          row.get("id", ""),
                "title":       row.get("title", "Untitled"),
                "rating":      rating,
                "tags":        tags,
                "tags_str":    ", ".join(tags),
                "url":         url,
                "search_text": f"{row.get('title','')} {' '.join(tags)}",
            })
    return problems


# ── COSINE SIMILARITY ──────────────────────────────────────────────────────
def cosine_sim(a, b) -> float:
    if NUMPY:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── ANTHROPIC VOYAGE EMBEDDINGS ────────────────────────────────────────────
def get_embedding(text: str) -> list:
    """Get a single embedding from Anthropic Voyage API."""
    import subprocess, tempfile
    payload = json.dumps({
        "model": "voyage-3",
        "input": [text],
        "input_type": "query"
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(payload)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "https://api.voyageai.com/v1/embeddings",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {ANTHROPIC_API_KEY}",
                "-d", f"@{tmp_path}",
            ],
            capture_output=True, text=True, timeout=30
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["data"][0]["embedding"]


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts (max 128 per call)."""
    import subprocess, tempfile
    payload = json.dumps({
        "model": "voyage-3",
        "input": texts,
        "input_type": "document"
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(payload)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "https://api.voyageai.com/v1/embeddings",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {ANTHROPIC_API_KEY}",
                "-d", f"@{tmp_path}",
            ],
            capture_output=True, text=True, timeout=120
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


def build_embeddings(problems: list) -> list:
    """Build and cache embeddings for all problems."""
    print(f"Building embeddings for {len(problems):,} problems using Anthropic Voyage...")
    print("This will take a few minutes. Embeddings are cached after first run.")
    embeddings = []
    batch_size = 32  # smaller batch to stay within Voyage free tier limits
    total = len(problems)
    for i in range(0, total, batch_size):
        batch = problems[i:i + batch_size]
        texts = [p["search_text"] for p in batch]
        # retry up to 3 times on failure
        for attempt in range(3):
            try:
                vecs = get_embeddings_batch(texts)
                embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retrying batch {i}-{i+batch_size} (attempt {attempt+2}/3)...")
                    time.sleep(3)
                else:
                    print(f"  ERROR on batch {i}-{i+batch_size}: {e}")
                    embeddings.extend([[0.0] * 1024] * len(batch))
        pct = min(i + batch_size, total)
        print(f"  Embedded {pct:,}/{total:,} problems...", end="\r")
        time.sleep(1.0)  # longer pause to respect rate limits
    print(f"\nDone! Saving to {CACHE_PATH}")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


def load_or_build_embeddings(problems: list) -> list:
    """Load cached embeddings or build them if missing."""
    if Path(CACHE_PATH).exists():
        print(f"Loading cached embeddings from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(problems):
            print(f"Loaded {len(embeddings):,} embeddings from cache.")
            return embeddings
        else:
            print("Cache size mismatch — rebuilding embeddings...")
    return build_embeddings(problems)


# ── SEMANTIC SEARCH ────────────────────────────────────────────────────────
def semantic_search(query: str, problems: list, embeddings: list,
                    min_rating: int, max_rating: int,
                    tag_filter: str, top_k: int) -> list:
    """Search by meaning using cosine similarity of embeddings."""
    query_vec = get_embedding(query)
    results = []
    for i, p in enumerate(problems):
        r = p["rating"]
        if r and not (min_rating <= r <= max_rating):
            continue
        if tag_filter and tag_filter.lower() not in p["tags_str"].lower():
            continue
        score = cosine_sim(query_vec, embeddings[i])
        results.append({**p, "score": score})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── KEYWORD SEARCH ─────────────────────────────────────────────────────────
def keyword_search(query: str, problems: list,
                   min_rating: int, max_rating: int,
                   tag_filter: str, top_k: int) -> list:
    tokens = re.findall(r'\w+', query.lower())
    results = []
    for p in problems:
        r = p["rating"]
        if r and not (min_rating <= r <= max_rating):
            continue
        if tag_filter and tag_filter.lower() not in p["tags_str"].lower():
            continue
        haystack = p["search_text"].lower()
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            results.append({**p, "score": score / len(tokens)})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── GROQ — AI SOLUTION ─────────────────────────────────────────────────────
def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    import subprocess, tempfile
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    system_msg = (
        "You are an expert competitive programmer who has solved thousands of Codeforces problems. "
        "Your task is to provide a CORRECT, COMPLETE, and DIRECTLY SUBMITTABLE C++ solution. "
        "Rules:\n"
        "- The C++ code MUST compile with g++ and pass ALL test cases on Codeforces when submitted.\n"
        "- Use fast I/O: ios_base::sync_with_stdio(false); cin.tie(NULL);\n"
        "- Handle ALL edge cases (n=0, n=1, large inputs, negative numbers, overflow, etc.).\n"
        "- Use long long when there is any risk of integer overflow.\n"
        "- Do NOT write placeholder code or pseudocode — only real, working C++.\n"
        "- If you are not 100% sure of the solution, write the most likely correct approach anyway.\n"
        "- The ```cpp code block must contain a complete program with #include and int main()."
    )
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(payload)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "https://api.groq.com/openai/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {GROQ_API_KEY}",
                "-d", f"@{tmp_path}",
            ],
            capture_output=True, text=True, timeout=60
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    data = json.loads(result.stdout)
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]


# ── HTML UI ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CF Search</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #080b10; --surface: #0e1420; --border: #162030;
    --accent: #3b9eff; --accent2: #00e5a0;
    --text: #cdd8e8; --muted: #4a6080; --tag-bg: #0d1a2a;
    --tag-text: #4a9edd; --card-hover: #111c2e;
    --rating-low: #00e5a0; --rating-mid: #ffd060; --rating-hard: #ff5060;
  }
  body { font-family:'Sora',sans-serif; background:var(--bg); color:var(--text);
         min-height:100vh; display:flex; flex-direction:column; }
  header { padding: 2.5rem 2rem 1.5rem; border-bottom:1px solid var(--border); }
  .logo { font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:var(--accent2);
          letter-spacing:0.08em; margin-bottom:0.4rem; }
  h1 { font-size:2rem; font-weight:600; color:var(--text); line-height:1.2; }
  h1 span { color:var(--accent); }
  .search-bar { display:flex; gap:0.75rem; margin-top:1.5rem; flex-wrap:wrap; }
  .search-bar input[type=text], .search-bar input[type=number] {
    background:var(--surface); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:0.65rem 1rem; font-family:'Sora',sans-serif;
    font-size:0.95rem; outline:none; transition:border 0.2s;
  }
  .search-bar input[type=text]:focus { border-color:var(--accent); }
  #query { flex:1; min-width:220px; }
  .rating-group { display:flex; align-items:center; gap:0.4rem; }
  .rating-group input { width:80px; }
  .rating-group span { color:var(--muted); font-size:0.85rem; }
  .mode-toggle { display:flex; align-items:center; gap:0.5rem; }
  .mode-toggle label { font-size:0.85rem; color:var(--muted); }
  .mode-toggle select {
    background:var(--surface); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:0.5rem 0.75rem; font-family:'Sora',sans-serif;
    font-size:0.85rem; outline:none; cursor:pointer;
  }
  button { background:var(--accent); border:none; color:#fff; border-radius:8px;
           padding:0.65rem 1.4rem; font-family:'Sora',sans-serif; font-size:0.95rem;
           font-weight:500; cursor:pointer; transition:opacity 0.15s; white-space:nowrap; }
  button:hover { opacity:0.85; }
  button:disabled { opacity:0.5; cursor:default; }
  main { flex:1; padding:1.5rem 2rem; max-width:1100px; width:100%; margin:0 auto; }
  #status { color:var(--muted); font-size:0.88rem; margin-bottom:1rem;
            font-family:'JetBrains Mono',monospace; }
  #results { display:flex; flex-direction:column; gap:0.75rem; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px;
          padding:1.1rem 1.4rem; transition:background 0.15s, border-color 0.15s; }
  .card:hover { background:var(--card-hover); border-color:#2a3050; }
  .card-title { font-weight:500; font-size:1rem; color:var(--text); text-decoration:none; }
  .card-title:hover { color:var(--accent); }
  .card-meta { display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-top:0.55rem; }
  .rating-badge { font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:500;
                  padding:2px 8px; border-radius:4px; }
  .r-low  { background:rgba(61,255,160,0.12); color:var(--rating-low); }
  .r-mid  { background:rgba(240,192,64,0.12);  color:var(--rating-mid); }
  .r-hard { background:rgba(240,90,90,0.12);   color:var(--rating-hard); }
  .r-none { background:rgba(107,114,128,0.15); color:var(--muted); }
  .score-badge { font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                 color:var(--accent2); opacity:0.7; }
  .mode-badge { font-size:0.72rem; padding:2px 7px; border-radius:4px;
                background:rgba(91,138,240,0.15); color:var(--accent); }
  .tags { display:flex; flex-wrap:wrap; gap:0.35rem; }
  .tag { background:var(--tag-bg); color:var(--tag-text); border-radius:4px;
         padding:2px 7px; font-size:0.75rem; }
  .empty { text-align:center; padding:3rem 1rem; color:var(--muted); }
  .empty b { display:block; font-size:1.1rem; color:var(--text); margin-bottom:0.5rem; }
  .spinner { display:inline-block; width:18px; height:18px; border:2px solid var(--border);
             border-top-color:var(--accent); border-radius:50%; animation:spin 0.7s linear infinite;
             vertical-align:middle; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .solution-toggle { margin-top:0.75rem; display:flex; align-items:center; gap:0.5rem; cursor:pointer; }
  .solution-toggle input[type=checkbox] { accent-color:var(--accent); width:15px; height:15px; cursor:pointer; }
  .solution-toggle label { font-size:0.82rem; color:var(--muted); cursor:pointer; user-select:none; }
  .solution-toggle label:hover { color:var(--accent); }
  .solution-box { margin-top:0.75rem; background:#0a0c11; border:1px solid var(--border);
                  border-radius:8px; padding:1.1rem 1.3rem; font-size:0.87rem; line-height:1.65;
                  color:#c9cdd8; display:none; word-break:break-word; }
  .solution-box.open { display:block; }
  .solution-box code, .solution-box pre { font-family:'JetBrains Mono',monospace; font-size:0.82rem; }
  .solution-box pre { background:#060709; border:1px solid #1e2130; border-radius:6px;
                      padding:0.8rem 1rem; overflow-x:auto; margin-top:0.5rem; white-space:pre; }
  .solution-box b { color:var(--accent2); }
  .sol-loading { color:var(--muted); font-style:italic; }
  .banner { border-radius:10px; padding:1rem 1.3rem; margin-bottom:1rem;
            font-size:0.88rem; line-height:1.7; }
  .banner code { font-family:'JetBrains Mono',monospace; background:rgba(255,255,255,0.07);
                 border-radius:4px; padding:1px 6px; }
  .banner a { color:var(--accent); }
  .banner-warn { background:rgba(240,90,90,0.08); border:1px solid rgba(240,90,90,0.3); }
  .banner-info { background:rgba(91,138,240,0.08); border:1px solid rgba(91,138,240,0.3); }
</style>
</head>
<body>
<header>
  <div class="logo">&lt;/&gt; Codeforces search</div>
  <h1>Find your next <span>problem</span></h1>
  <div class="search-bar">
    <input id="query" type="text" placeholder="Describe the problem… e.g. shortest path in grid with obstacles" autocomplete="off">

    <div class="rating-group">
      <span style="color:var(--muted);font-size:0.85rem;">Rating:</span>
      <input id="min-r" type="number" placeholder="min ★" value="" min="0" max="4000">
      <span>–</span>
      <input id="max-r" type="number" placeholder="max ★" value="" min="0" max="4000">
    </div>
    <input id="topk" type="number" value="10" min="1" max="50" style="width:72px" title="Results count">
    <div class="mode-toggle">
      <select id="search-mode">
        <option value="semantic" selected>Semantic only</option>
        <option value="keyword"> Keyword only</option>
      </select>
    </div>
    <button id="search-btn" onclick="doSearch()">Search</button>
  </div>
</header>
<main>
  <div id="banner-groq" class="banner banner-warn" style="display:none">
    <b>⚠️ GROQ_API_KEY not set — AI solutions won't work.</b><br>
    Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>, then run:
    <code>set GROQ_API_KEY=your-key-here</code> (Windows) or <code>export GROQ_API_KEY="your-key-here"</code> (Mac/Linux)
  </div>
  <div id="banner-anthropic" class="banner banner-info" style="display:none">
    <b>💡 ANTHROPIC_API_KEY not set — Semantic search unavailable, using keyword search.</b><br>
    Get a key at <a href="https://console.anthropic.com" target="_blank">console.anthropic.com</a>, then run:
    <code>set ANTHROPIC_API_KEY=your-key-here</code> (Windows) or <code>export ANTHROPIC_API_KEY="your-key-here"</code> (Mac/Linux)
  </div>
  <div id="status">Ready — type a query and press Search.</div>
  <div id="results"></div>
</main>

<script>
const HAS_GROQ_KEY      = __HAS_GROQ_KEY__;
const HAS_ANTHROPIC_KEY = __HAS_ANTHROPIC_KEY__;
const EMBEDDINGS_READY  = __EMBEDDINGS_READY__;

if (!HAS_GROQ_KEY)      document.getElementById('banner-groq').style.display = 'block';
if (!HAS_ANTHROPIC_KEY) document.getElementById('banner-anthropic').style.display = 'block';

// Disable semantic options if no embeddings
if (!EMBEDDINGS_READY) {
  document.querySelectorAll('#search-mode option').forEach(o => {
    if (o.value === 'semantic') o.disabled = true;
    if (o.value === 'auto') o.text = '🔑 Keyword (no embeddings)';
  });
}

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

function ratingClass(r) {
  if (!r) return 'r-none';
  if (r <= 1400) return 'r-low';
  if (r <= 2000) return 'r-mid';
  return 'r-hard';
}

function renderResults(data) {
  const el = document.getElementById('results');
  if (!data.results || data.results.length === 0) {
    el.innerHTML = '<div class="empty"><b>No results found</b>Try different keywords or remove filters.</div>';
    return;
  }
  const modeLabel = data.mode === 'semantic' ? 'semantic search' : '🔑 keyword search';
  document.getElementById('status').textContent =
    `${data.results.length} result${data.results.length !== 1 ? 's' : ''} · ${modeLabel} · ${data.time_ms}ms`;

  el.innerHTML = data.results.map((p, i) => {
    const rc = ratingClass(p.rating);
    const rLabel = p.rating ? `★ ${p.rating}` : '★ N/A';
    const tags = (p.tags || []).slice(0, 6).map(t => `<span class="tag">${t}</span>`).join('');
    const scoreStr = data.mode === 'semantic'
      ? `<span class="score-badge">${(p.score * 100).toFixed(1)}% match</span>` : '';
    const modeBadge = `<span class="mode-badge">${data.mode === 'semantic' ? 'semantic' : '🔑 keyword'}</span>`;
    return `<div class="card">
      <a class="card-title" href="${p.url}" target="_blank" rel="noopener">${i+1}. ${p.title}</a>
      <div class="card-meta">
        <span class="rating-badge ${rc}">${rLabel}</span>
        ${scoreStr}
        ${modeBadge}
        <div class="tags">${tags}</div>
      </div>
      <div class="solution-toggle">
        <input type="checkbox" id="chk-${i}"
          data-title="${p.title.replace(/"/g,'&quot;')}"
          data-tags="${p.tags_str.replace(/"/g,'&quot;')}"
          data-url="${(p.url||'').replace(/"/g,'&quot;')}"
          onchange="toggleSolution(this, this.dataset.title, this.dataset.tags, this.dataset.url, 'sol-${i}')">
        <label for="chk-${i}">💡 Show AI solution (Groq)</label>
      </div>
      <div class="solution-box" id="sol-${i}"></div>
    </div>`;
  }).join('');
}

async function toggleSolution(checkbox, title, tags, url, boxId) {
  const box = document.getElementById(boxId);
  if (!checkbox.checked) { box.classList.remove('open'); return; }
  if (box.dataset.loaded === '1') { box.classList.add('open'); return; }
  box.innerHTML = '<span class="sol-loading"><span class="spinner"></span> Groq is thinking…</span>';
  box.classList.add('open');
  try {
    const resp = await fetch('/api/solution?' + new URLSearchParams({ title, tags, url }));
    const data = await resp.json();
    if (data.error) {
      box.innerHTML = '❌ Error: ' + data.error;
    } else {
      box.innerHTML = formatSolution(data.solution);
      box.dataset.loaded = '1';
    }
  } catch(e) {
    box.innerHTML = '❌ Server error: ' + e.message;
  }
}

function formatSolution(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```cpp([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\n/g, '<br>');
}

async function doSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const btn = document.getElementById('search-btn');
  btn.disabled = true;
  document.getElementById('status').innerHTML = '<span class="spinner"></span>Searching…';
  document.getElementById('results').innerHTML = '';

  const params = new URLSearchParams({
    q:     query,
    min_r: document.getElementById('min-r').value || 0,
    max_r: document.getElementById('max-r').value || 9999,
    top_k: document.getElementById('topk').value || 10,
    mode:  document.getElementById('search-mode').value,
  });

  try {
    const resp = await fetch('/api/search?' + params);
    const data = await resp.json();
    if (data.error) {
      document.getElementById('status').textContent = 'Error: ' + data.error;
    } else {
      renderResults(data);
    }
  } catch(e) {
    document.getElementById('status').textContent = 'Server error: ' + e.message;
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""

# ── HTTP SERVER ────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    problems   = []
    embeddings = []
    html       = ""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(200, "text/html", Handler.html.encode())
        elif parsed.path == "/api/search":
            self._handle_search(parsed.query)
        elif parsed.path == "/api/solution":
            self._handle_solution(parsed.query)
        else:
            self._send(404, "text/plain", b"Not found")

    def _handle_search(self, qs: str):
        params = urllib.parse.parse_qs(qs)
        query  = (params.get("q",     [""])[0]).strip()
        min_r  = int(params.get("min_r", [0])[0] or 0)
        max_r  = int(params.get("max_r", [9999])[0] or 9999)
        tag    = params.get("tag",   [""])[0].strip()
        top_k  = min(int(params.get("top_k", [10])[0] or 10), 50)
        mode   = params.get("mode",  ["auto"])[0]

        t0 = time.time()
        result_mode = "keyword"
        try:
            if mode == "semantic" and Handler.embeddings and ANTHROPIC_API_KEY:
                results = semantic_search(query, Handler.problems, Handler.embeddings,
                                          min_r, max_r, tag, top_k)
                result_mode = "semantic"
            elif mode == "keyword":
                results = keyword_search(query, Handler.problems, min_r, max_r, tag, top_k)
            else:
                # Auto: try semantic first, fall back to keyword if weak or unavailable
                if Handler.embeddings and ANTHROPIC_API_KEY:
                    try:
                        results = semantic_search(query, Handler.problems, Handler.embeddings,
                                                  min_r, max_r, tag, top_k)
                        # Fall back to keyword if top score is very low
                        if not results or results[0]["score"] < 0.3:
                            kw = keyword_search(query, Handler.problems, min_r, max_r, tag, top_k)
                            if kw:
                                results = kw
                                result_mode = "keyword"
                            else:
                                result_mode = "semantic"
                        else:
                            result_mode = "semantic"
                    except Exception as e:
                        print(f"[SEARCH] Semantic failed, falling back to keyword: {e}")
                        results = keyword_search(query, Handler.problems, min_r, max_r, tag, top_k)
                else:
                    results = keyword_search(query, Handler.problems, min_r, max_r, tag, top_k)

        except Exception as e:
            self._send(200, "application/json", json.dumps({"error": str(e)}).encode())
            return

        elapsed = int((time.time() - t0) * 1000)
        self._send(200, "application/json",
                   json.dumps({"results": results, "mode": result_mode, "time_ms": elapsed},
                              ensure_ascii=False).encode())

    def _handle_solution(self, qs: str):
        params = urllib.parse.parse_qs(qs)
        title  = (params.get("title", [""])[0]).strip()
        tags   = (params.get("tags",  [""])[0]).strip()
        # url passed from frontend so the model knows the exact problem
        print(f"[AI] Solution requested for: {title}")
        url = (params.get("url", [""])[0]).strip()
        url_line = f"Problem URL: {url}\n" if url else ""
        prompt = (
            f"Solve the following Codeforces competitive programming problem.\n\n"
            f"Problem title: \"{title}\"\n"
            f"{url_line}"
            f"Tags (algorithmic topics): {tags}\n\n"
            f"Based on the title and tags, reason carefully about what the problem is asking "
            f"and what the correct algorithm is. Then provide your answer in EXACTLY these three sections:\n\n"
            f"**Approach** — Identify the algorithm from the tags. Explain the key insight and why this approach works. (3-5 sentences)\n\n"
            f"**Step-by-step** — Numbered implementation steps with specific details (data structures, complexity, edge cases).\n\n"
            f"**C++ Code** — A COMPLETE, CORRECT, DIRECTLY SUBMITTABLE C++ solution:\n"
            f"- Must include all necessary #include headers\n"
            f"- Must have int main() with fast I/O (ios_base::sync_with_stdio(false); cin.tie(NULL);)\n"
            f"- Must handle all edge cases\n"
            f"- Must use long long where overflow is possible\n"
            f"- Must be wrapped in ```cpp ... ``` block\n\n"
            f"IMPORTANT: Write the actual correct solution for this specific problem based on its tags and title. "
            f"Do not write generic template code."
        )
        try:
            print("[AI] Calling Groq...")
            text = call_groq(prompt)
            print(f"[AI] Success! {len(text)} chars")
            self._send(200, "application/json", json.dumps({"solution": text}).encode())
        except Exception as e:
            import traceback
            print(f"[AI] ERROR: {e}")
            traceback.print_exc()
            self._send(200, "application/json", json.dumps({"error": str(e)}).encode())

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codeforces Semantic Search")
    parser.add_argument("--dataset", default="CodeForces.csv")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--embed", action="store_true",
                        help="Force rebuild embeddings even if cache exists")
    args = parser.parse_args()

    print("Loading dataset…")
    problems = load_dataset(args.dataset)
    print(f"Loaded {len(problems):,} problems from '{args.dataset}'")

    # Status
    if GROQ_API_KEY:
        print("✅ GROQ_API_KEY found — AI solutions enabled")
    else:
        print("⚠️  GROQ_API_KEY not set — AI solutions disabled")

    embeddings = []
    if ANTHROPIC_API_KEY:
        print("✅ ANTHROPIC_API_KEY found — Semantic search enabled")
        if args.embed and Path(CACHE_PATH).exists():
            Path(CACHE_PATH).unlink()
            print("Deleted old cache, rebuilding…")
        embeddings = load_or_build_embeddings(problems)
    else:
        print("⚠️  ANTHROPIC_API_KEY not set — Semantic search disabled, using keyword only")
        print("   Get a key at https://console.anthropic.com")
        print("   Then run: set ANTHROPIC_API_KEY=your-key-here")

    embeddings_ready = len(embeddings) == len(problems) and len(embeddings) > 0

    html = HTML \
        .replace("__HAS_GROQ_KEY__",      "true" if GROQ_API_KEY else "false") \
        .replace("__HAS_ANTHROPIC_KEY__",  "true" if ANTHROPIC_API_KEY else "false") \
        .replace("__EMBEDDINGS_READY__",   "true" if embeddings_ready else "false")

    Handler.problems   = problems
    Handler.embeddings = embeddings
    Handler.html       = html

    server = http.server.HTTPServer(("", args.port), Handler)
    url    = f"http://localhost:{args.port}"
    print(f"\nServer running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
