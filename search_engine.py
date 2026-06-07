"""
Codeforces Semantic Search Engine
Run: python search_engine.py
Then open: http://localhost:5000

AI Solutions powered by Groq (free tier, no credit card needed).
Get your free key at: https://console.groq.com
Then run:
  set GROQ_API_KEY=your-key-here        (Windows)
  export GROQ_API_KEY="your-key-here"   (Mac/Linux)
  python search_engine.py --dataset CodeForces.csv
"""

import os, json, pickle, re, csv, math, http.server, threading, urllib.parse, webbrowser
from pathlib import Path

# ── Optional numpy for faster cosine similarity ──────────────────────────────
try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False

CACHE_PATH   = "cf_embeddings.pkl"

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
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
                "id":       row.get("id", ""),
                "title":    row.get("title", "Untitled"),
                "rating":   rating,
                "tags":     tags,
                "tags_str": ", ".join(tags),
                "url":      url,
                "search_text": f"{row.get('title','')} {' '.join(tags)}",
            })
    return problems


# ─────────────────────────────────────────────────────────────────────────────
#  GROQ — AI SOLUTION (free API, no payment needed)
#  Get your free key at: https://console.groq.com
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq free API using curl subprocess to avoid urllib header issues."""
    import subprocess, tempfile

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "then run:  set GROQ_API_KEY=your-key-here  (Windows) "
            "or  export GROQ_API_KEY=your-key-here  (Mac/Linux)"
        )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
    })

    # Write payload to temp file to avoid command-line escaping issues on Windows
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


# ─────────────────────────────────────────────────────────────────────────────
#  COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    if NUMPY:
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD SEARCH  (always available, no API needed)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_search(query: str, problems: list[dict],
                   min_rating: int, max_rating: int,
                   tag_filter: str, top_k: int) -> list[dict]:
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


# ─────────────────────────────────────────────────────────────────────────────
#  HTML UI
# ─────────────────────────────────────────────────────────────────────────────

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
    --bg: #0d0f14; --surface: #13151c; --border: #1e2130;
    --accent: #5b8af0; --accent2: #3dffa0;
    --text: #e8eaf0; --muted: #6b7280; --tag-bg: #1a2035;
    --tag-text: #7aa0f5; --card-hover: #181b28;
    --rating-low: #3dffa0; --rating-mid: #f0c040; --rating-hard: #f05a5a;
  }
  body { font-family:'Sora',sans-serif; background:var(--bg); color:var(--text);
         min-height:100vh; display:flex; flex-direction:column; }
  header { padding: 2.5rem 2rem 1.5rem; border-bottom:1px solid var(--border); }
  .logo { font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:var(--accent2);
          letter-spacing:0.08em; margin-bottom:0.4rem; }
  h1 { font-size:2rem; font-weight:600; color:var(--text); line-height:1.2; }
  h1 span { color:var(--accent); }

  .search-bar { display:flex; gap:0.75rem; margin-top:1.5rem; flex-wrap:wrap; }
  .search-bar input[type=text], .search-bar select, .search-bar input[type=number] {
    background:var(--surface); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:0.65rem 1rem; font-family:'Sora',sans-serif;
    font-size:0.95rem; outline:none; transition:border 0.2s;
  }
  .search-bar input[type=text]:focus { border-color:var(--accent); }
  #query { flex:1; min-width:220px; }
  .rating-group { display:flex; align-items:center; gap:0.4rem; }
  .rating-group input { width:80px; }
  .rating-group span { color:var(--muted); font-size:0.85rem; }
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

  .no-key-banner { background:rgba(240,90,90,0.08); border:1px solid rgba(240,90,90,0.3);
                   border-radius:10px; padding:1rem 1.3rem; margin-bottom:1.5rem; font-size:0.88rem; line-height:1.7; }
  .no-key-banner code { font-family:'JetBrains Mono',monospace; background:rgba(255,255,255,0.07);
                        border-radius:4px; padding:1px 6px; }
  .no-key-banner a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <div class="logo">&lt;/&gt; codeforces search</div>
  <h1>Find your next <span>problem</span></h1>
  <div class="search-bar">
    <input id="query" type="text" placeholder="Describe the problem… e.g. shortest path in grid with obstacles" autocomplete="off">
    <input id="tag-input" type="text" placeholder="tag filter" style="width:140px">
    <div class="rating-group">
      <input id="min-r" type="number" placeholder="min ★" value="" min="0" max="4000">
      <span>–</span>
      <input id="max-r" type="number" placeholder="max ★" value="" min="0" max="4000">
    </div>
    <input id="topk" type="number" value="10" min="1" max="50" style="width:72px" title="Results count">
    <button id="search-btn" onclick="doSearch()">Search</button>
  </div>
</header>
<main>
  <div id="no-key-banner" class="no-key-banner" style="display:none">
    <b>⚠️ GROQ_API_KEY not set — AI solutions won't work.</b><br>
    Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>, then run:
    <code>set GROQ_API_KEY=your-key-here</code> (Windows) or
    <code>export GROQ_API_KEY=your-key-here</code> (Mac/Linux)
  </div>
  <div id="status">Ready — type a query and press Search.</div>
  <div id="results"></div>
</main>

<script>
const HAS_GROQ_KEY = __HAS_GEMINI_KEY__;
if (!HAS_GROQ_KEY) {
  document.getElementById('no-key-banner').style.display = 'block';
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
  document.getElementById('status').textContent =
    `${data.results.length} result${data.results.length!==1?'s':''} · keyword search · ${data.time_ms}ms`;

  el.innerHTML = data.results.map((p, i) => {
    const rc = ratingClass(p.rating);
    const rLabel = p.rating ? `★ ${p.rating}` : '★ N/A';
    const tags = (p.tags || []).slice(0, 6).map(t => `<span class="tag">${t}</span>`).join('');
    return `<div class="card">
      <a class="card-title" href="${p.url}" target="_blank" rel="noopener">${i+1}. ${p.title}</a>
      <div class="card-meta">
        <span class="rating-badge ${rc}">${rLabel}</span>
        <div class="tags">${tags}</div>
      </div>
      <div class="solution-toggle">
        <input type="checkbox" id="chk-${i}" data-title="${p.title.replace(/"/g,'&quot;')}" data-tags="${p.tags_str.replace(/"/g,'&quot;')}"
          onchange="toggleSolution(this, this.dataset.title, this.dataset.tags, 'sol-${i}')">
        <label for="chk-${i}">💡 Show AI solution (Groq)</label>
      </div>
      <div class="solution-box" id="sol-${i}"></div>
    </div>`;
  }).join('');
}

async function toggleSolution(checkbox, title, tags, boxId) {
  const box = document.getElementById(boxId);
  if (!checkbox.checked) {
    box.classList.remove('open');
    return;
  }
  
  // Already loaded — just show it again
  if (box.dataset.loaded === '1') {
    box.classList.add('open');
    return;
  }
  box.innerHTML = '<span class="sol-loading"><span class="spinner"></span> Groq is thinking…</span>';
  box.classList.add('open');
  try {
    const params = new URLSearchParams({ title, tags });
    const resp = await fetch('/api/solution?' + params);
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
    tag:   document.getElementById('tag-input').value.trim(),
    top_k: document.getElementById('topk').value || 10,
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

# ─────────────────────────────────────────────────────────────────────────────
#  HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    problems    = []

    def log_message(self, fmt, *args):
        pass  # suppress access logs

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
        import time
        params = urllib.parse.parse_qs(qs)
        query  = (params.get("q",     [""])[0]).strip()
        min_r  = int(params.get("min_r", [0])[0] or 0)
        max_r  = int(params.get("max_r", [9999])[0] or 9999)
        tag    = params.get("tag",   [""])[0].strip()
        top_k  = min(int(params.get("top_k", [10])[0] or 10), 50)

        t0 = time.time()
        try:
            results = keyword_search(query, self.problems, min_r, max_r, tag, top_k)
        except Exception as e:
            self._send(200, "application/json", json.dumps({"error": str(e)}).encode())
            return

        elapsed = int((time.time() - t0) * 1000)
        self._send(200, "application/json",
                   json.dumps({"results": results, "mode": "keyword", "time_ms": elapsed},
                              ensure_ascii=False).encode())

    def _handle_solution(self, qs: str):
        params = urllib.parse.parse_qs(qs)
        title  = (params.get("title", [""])[0]).strip()
        tags   = (params.get("tags",  [""])[0]).strip()

        print(f"[AI] Solution requested for: {title}")

        
        prompt = (
            f"You are an expert competitive programmer. "
            f"Give a clear solution for the Codeforces problem titled \"{title}\" "
            f"with tags: {tags}.\n\n"
            f"Structure your answer in exactly these three sections:\n"
            f"1. **Approach** — explain the algorithm/idea in 3-5 sentences.\n"
            f"2. **Step-by-step** — numbered steps of the solution logic.\n"
            f"3. **C++ Code** — a clean, correct C++ solution inside a ```cpp block.\n"
            f"Be concise and precise."
        )

        try:
            print(f"[AI] Calling Groq...")
            text = call_groq(prompt)
            print(f"[AI] Success! Response length: {len(text)} chars")
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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codeforces Semantic Search")
    parser.add_argument("--dataset", default="CodeForces.csv",
                        help="Path to the Codeforces CSV file")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print("Loading dataset…")
    problems = load_dataset(args.dataset)
    print(f"Loaded {len(problems):,} problems from '{args.dataset}'")

    if GROQ_API_KEY:
        print("✅ GROQ_API_KEY found — AI solutions enabled (llama-3.3-70b-versatile)")
    else:
        print("⚠️  GROQ_API_KEY not set — AI solutions disabled.")
        print("   Get a free key at https://console.groq.com")
        print("   Then run: set GROQ_API_KEY=your-key-here")

    html = HTML.replace("__HAS_GEMINI_KEY__", "true" if GROQ_API_KEY else "false")

    Handler.problems = problems
    Handler.html     = html

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
