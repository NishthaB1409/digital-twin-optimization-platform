"""HTTP surface for the live control loop, plus a page to watch it on.

State is pushed with Server-Sent Events rather than polled. SSE is one long
GET over ordinary HTTP -- no extra dependency, no protocol upgrade, and it
survives corporate proxies that block WebSockets, which matters if this is ever
meant to run inside a plant network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .live import LiveConfig, LiveFactory

router = APIRouter(prefix="/live", tags=["live"])

#: One run at a time. A plant has one floor; this mirrors that rather than
#: pretending to be a multi-tenant service it is not.
_factory: LiveFactory | None = None


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str = Field("ppo", description="Policy to steer with")
    seed: int = Field(1000, description="Which job set the floor runs")
    time_scale: float = Field(
        20.0, gt=0, le=500, description="Simulated hours per real second"
    )
    decision_interval: float = Field(
        8.0, gt=0, description="Simulated hours between policy calls"
    )


def get_factory() -> LiveFactory:
    if _factory is None:
        raise HTTPException(
            status_code=409, detail="no run in progress; POST /live/start first"
        )
    return _factory


@router.post("/start")
def start(request: StartRequest) -> dict[str, Any]:
    """Begin a live run, replacing any run already in progress."""
    global _factory
    from .app import get_config, named_policy

    if _factory is not None and _factory.running:
        _factory.stop()

    _factory = LiveFactory(
        config=get_config(),
        policy=named_policy(request.policy),
        live_config=LiveConfig(
            time_scale=request.time_scale,
            decision_interval=request.decision_interval,
            seed=request.seed,
        ),
    )
    _factory.start()
    return {"started": True, "policy": request.policy, "seed": request.seed}


@router.post("/stop")
def stop() -> dict[str, Any]:
    factory = get_factory()
    factory.stop()
    return {"stopped": True, "clock_hours": factory.snapshot()["clock_hours"]}


@router.get("/state")
def state() -> dict[str, Any]:
    """Current floor state. Use /live/stream to be pushed it instead."""
    return get_factory().snapshot()


@router.get("/stream")
async def stream() -> StreamingResponse:
    """Server-Sent Events: one message per update until the run ends."""
    factory = get_factory()

    async def events():
        while True:
            snapshot = factory.snapshot()
            yield f"data: {json.dumps(snapshot)}\n\n"
            if not snapshot["running"]:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("", response_class=HTMLResponse)
def page() -> HTMLResponse:
    """A page that watches the floor."""
    return HTMLResponse(_PAGE)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factory Floor — Live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg:#f2f4f5; --card:#fff; --sunk:#e9eced; --ink:#14181b; --ink2:#5a646c;
  --line:#d5dadd; --accent:#0d5c6e; --ok:#2c6b4e; --warn:#9c6212; --bar:#2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0e1113; --card:#171b1e; --sunk:#1f2427; --ink:#e4e8ea; --ink2:#949fa6;
    --line:#282f33; --accent:#4fb0c6; --ok:#5cb68a; --warn:#d19a3c; --bar:#3987e5;
  }
}
*{box-sizing:border-box}
body{margin:0;padding:0 22px 60px;background:var(--bg);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6}
.wrap{max-width:940px;margin:0 auto}
header{padding:34px 0 18px;border-bottom:2px solid var(--ink)}
h1{font-size:26px;font-weight:600;letter-spacing:-.02em;margin:0 0 4px}
.sub{color:var(--ink2);margin:0;font-size:14px}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 6px;align-items:center}
button,select{font-family:"IBM Plex Mono",monospace;font-size:12px;padding:8px 14px;
  border:1px solid var(--line);background:var(--sunk);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#status{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  border:1px solid var(--line);margin-top:18px}
.tile{padding:12px 16px;border-right:1px solid var(--line)}
.tile:last-child{border-right:none}
.k{display:block;font-family:"IBM Plex Mono",monospace;font-size:9.5px;
  letter-spacing:.11em;text-transform:uppercase;color:var(--ink2);margin-bottom:4px}
.v{display:block;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.panel{border:1px solid var(--line);background:var(--card);margin-top:20px}
.panel h2{font-size:12px;font-family:"IBM Plex Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink2);margin:0;padding:12px 18px;
  border-bottom:1px solid var(--line);font-weight:500}
.row{display:grid;grid-template-columns:150px 1fr 92px;gap:14px;align-items:center;
  padding:9px 18px;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:none}
.nm{font-size:13.5px}
.track{background:var(--sunk);height:16px;position:relative;overflow:hidden}
.fill{background:var(--bar);height:100%;width:0;transition:width .25s linear}
.busy{position:absolute;inset:0;border-left:3px solid var(--ok);width:0;
  transition:width .25s linear}
.qn{font-family:"IBM Plex Mono",monospace;font-size:12px;text-align:right;
  color:var(--ink2);font-variant-numeric:tabular-nums}
.wrow{display:grid;grid-template-columns:150px 1fr 76px;gap:14px;align-items:center;
  padding:9px 18px;border-bottom:1px solid var(--line)}
.wrow:last-child{border-bottom:none}
.wtrack{background:var(--sunk);height:14px;position:relative}
.wmid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line)}
.wfill{position:absolute;top:0;bottom:0;background:var(--accent);transition:all .3s}
.wv{font-family:"IBM Plex Mono",monospace;font-size:12px;text-align:right;
  font-variant-numeric:tabular-nums}
#log{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink2);
  padding:10px 18px;max-height:150px;overflow-y:auto}
#log div{padding:2px 0}
.tag-fallback{color:var(--warn)}
.tag-finish{color:var(--ok)}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Factory floor — live</h1>
    <p class="sub">The twin runs on a wall clock. Every simulated shift the policy reads the floor and rewrites the dispatch weights.</p>
  </header>

  <div class="controls">
    <select id="policy">
      <option value="ppo">policy: ppo (trained)</option>
      <option value="blend">policy: blend (tuned fixed)</option>
      <option value="spt">policy: spt</option>
      <option value="mwkr">policy: mwkr (deliberately bad)</option>
    </select>
    <select id="speed">
      <option value="10">10x</option>
      <option value="20" selected>20x</option>
      <option value="60">60x</option>
      <option value="200">200x</option>
    </select>
    <button id="go">Start</button>
    <button id="halt">Stop</button>
    <span id="status">idle</span>
  </div>

  <div class="tiles">
    <div class="tile"><span class="k">Sim clock</span><span class="v" id="clock">—</span></div>
    <div class="tile"><span class="k">Completed</span><span class="v" id="done">—</span></div>
    <div class="tile"><span class="k">On the floor</span><span class="v" id="wip">—</span></div>
    <div class="tile"><span class="k">On time</span><span class="v" id="ontime">—</span></div>
    <div class="tile"><span class="k">Decisions</span><span class="v" id="dec">—</span></div>
  </div>

  <div class="panel">
    <h2>Stations — queue depth and machines busy</h2>
    <div id="stations"></div>
  </div>

  <div class="panel">
    <h2>Dispatch weights the policy is currently applying</h2>
    <div id="weights"></div>
  </div>

  <div class="panel">
    <h2>Events</h2>
    <div id="log"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let src = null;

function render(s) {
  $('clock').textContent = s.clock_hours.toFixed(0) + 'h';
  $('done').textContent = s.jobs_completed + '/' + s.jobs_total;
  $('wip').textContent = s.jobs_in_progress;
  $('ontime').textContent = s.jobs_completed ? (s.on_time_rate * 100).toFixed(0) + '%' : '—';
  $('dec').textContent = s.decisions;
  $('status').textContent = s.finished ? 'finished'
    : s.running ? ('running · ' + s.policy + ' · ' + s.time_scale + 'x') : 'stopped';

  const maxQ = Math.max(6, ...s.stations.map(x => x.queue_length));
  $('stations').innerHTML = s.stations.map(x => `
    <div class="row">
      <span class="nm">${x.name}</span>
      <span class="track">
        <span class="fill" style="width:${(x.queue_length / maxQ * 100).toFixed(1)}%"></span>
        <span class="busy" style="width:${(x.busy_machines / x.capacity * 100).toFixed(1)}%"></span>
      </span>
      <span class="qn">${x.queue_length} q · ${x.busy_machines}/${x.capacity}</span>
    </div>`).join('');

  $('weights').innerHTML = Object.entries(s.weights).map(([k, v]) => {
    const pct = Math.abs(v) * 50;
    const left = v >= 0 ? 50 : 50 - pct;
    return `<div class="wrow">
      <span class="nm">${k.replace(/_/g, ' ')}</span>
      <span class="wtrack"><span class="wmid"></span>
        <span class="wfill" style="left:${left}%;width:${pct}%"></span></span>
      <span class="wv">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
    </div>`;
  }).join('');

  $('log').innerHTML = s.events.slice().reverse().map(e =>
    `<div><span class="tag-${e.kind}">${e.clock_hours.toFixed(0).padStart(4)}h  ${e.kind.padEnd(9)}</span> ${e.detail}</div>`
  ).join('');
}

function listen() {
  if (src) src.close();
  src = new EventSource('/live/stream');
  src.onmessage = ev => render(JSON.parse(ev.data));
  src.onerror = () => { $('status').textContent = 'stream closed'; src.close(); src = null; };
}

$('go').onclick = async () => {
  $('status').textContent = 'starting…';
  await fetch('/live/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      policy: $('policy').value,
      time_scale: parseFloat($('speed').value),
      seed: 1000
    })
  });
  listen();
};
$('halt').onclick = async () => {
  await fetch('/live/stop', {method: 'POST'});
  if (src) { src.close(); src = null; }
  $('status').textContent = 'stopped';
};
</script>
</body>
</html>
"""
