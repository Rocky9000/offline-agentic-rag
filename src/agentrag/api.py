"""FastAPI service with a minimal built-in web UI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .pipeline import AgenticRAG


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


def create_app(rag: AgenticRAG) -> FastAPI:
    app = FastAPI(title="Offline Agentic RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(rag.index), "embedder": rag.index.embedder.name}

    @app.post("/query")
    def query(body: QueryIn) -> dict:
        if len(rag.index) == 0:
            raise HTTPException(409, "index is empty; POST /documents or run `agentrag ingest` first")
        return rag.ask(body.question).to_dict()

    @app.post("/documents", status_code=201)
    def add_document(body: DocumentIn) -> dict:
        chunks = chunk_text(body.text, body.doc_id)
        rag.index.add(chunks)
        return {"doc_id": body.doc_id, "chunks_added": len(chunks), "total_chunks": len(rag.index)}

    @app.get("/", response_class=HTMLResponse)
    def ui() -> str:
        return _UI

    return app


_UI = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agentic RAG</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--accent:#0f766e}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--accent:#2dd4bf}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
form{display:flex;gap:8px}input{flex:1;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:16px}
.tag{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted);margin-right:6px}
.src{font-size:14px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px;margin-top:8px}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted)}
</style></head><body><main>
<h1>Offline Agentic RAG</h1><p class="sub">Planner → hybrid retrieval → answer → critic. Runs entirely on your machine.</p>
<form id="f"><input id="q" placeholder="Ask about your documents…" autocomplete="off"><button>Ask</button></form>
<div id="out"></div>
<script>
const out=document.getElementById('out');
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
 const q=document.getElementById('q').value.trim(); if(!q) return;
 out.innerHTML='<div class="card">Thinking…</div>';
 const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 const d=await r.json(); if(!r.ok){out.innerHTML='<div class="card">'+esc(d.detail||'error')+'</div>';return;}
 out.innerHTML='<div class="card"><span class="tag">'+(d.grounded?'grounded':'unverified')+'</span><span class="tag">'+d.iterations+' iteration(s)</span><span class="tag">'+d.latency_ms+' ms</span><p>'+esc(d.answer)+'</p>'+
 d.sources.map(s=>'<div class="src"><b>['+s.n+'] '+esc(s.doc_id)+'</b><br>'+esc(s.text.slice(0,280))+'…</div>').join('')+
 '<details><summary>Agent trace</summary><pre>'+esc(JSON.stringify(d.trace,null,2))+'</pre></details></div>';};
</script></main></body></html>"""
