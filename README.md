# Digital-Twin Smart Factory Optimization Platform

A SimPy digital twin of a 6-station aerospace manufacturing line, wrapped as a
Gymnasium environment, where a reinforcement-learning agent tunes the four
weights of a composite dispatching rule. Classical OR heuristics and provable
lower bounds are the benchmarks.

**255 tests · 3,700 lines of source · 2,084 lines of tests · containerised**

---

## The one idea

The project is **a factory simulator with a four-number knob on it.** Everything
else measures whether turning that knob well makes the factory run better.

When a machine frees up and several jobs are waiting, something must choose. The
choice is made by scoring every waiting job on four features — processing time,
slack, remaining work, and waiting time — and dispatching the lowest score. The
four numbers are how much each feature counts.

Set them to `[1, 0, 0, 0]` and you get "shortest job first" (SPT). The agent's
whole job is to pick those four numbers, and to change them as the floor
changes. It never picks a job directly.

That framing is deliberate. Choosing a job per station would give a
combinatorial action space that changes shape as queues grow; four continuous
weights give a fixed `Box(-1, 1, (4,))` that PPO handles natively — and the
learned policy stays readable, because you can see whether it favoured short
jobs or urgent ones.

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

Then work through these in order — each builds on what the last one showed:

```powershell
python scripts\run_simulation.py      # one factory run, KPIs, dispatch leverage
python scripts\run_simulation.py --compare   # seven rules, same jobs
python scripts\run_env.py             # policies scored through the RL env
python scripts\train_agent.py         # train PPO (~8 min)
python scripts\run_benchmark.py       # everything vs provable lower bounds
python scripts\make_dashboard.py      # → runs\dashboard.html
python scripts\serve.py --demo        # the API, in-process
```

If your prompt does not start with `(.venv)`, activation did not take and every
import will fail.

---

## Results

All figures below are **seed-paired**: every policy runs the same 16 job sets,
and differences are reported as paired means with win rates and p-values. This
is not optional here — see [Instance variance](#instance-variance-dominates).

### The agent reaches parity, and does not beat the best rules

| PPO vs | Difference | Win rate | p | Verdict |
|---|---:|---:|---:|---|
| `blend` | −0.96 | 44% | 0.652 | tie |
| `spt` | +1.72 | 50% | 0.558 | tie |
| `lwkr` | +27.81 | 94% | 0.001 | **beats** |
| `fifo` | +52.61 | 81% | 0.004 | **beats** |
| `min_slack` | +57.48 | 81% | 0.004 | **beats** |
| `lifo` | +89.92 | 100% | 0.000 | **beats** |
| `lpt` | +200.63 | 100% | 0.000 | **beats** |
| `mwkr` | +201.22 | 100% | 0.000 | **beats** |

The learned policy does vary its weights with the state (mean per-feature std
0.108), so it is a genuine state-dependent rule — just not a better one than a
well-tuned constant.

### How much room was ever there

| Metric | Best policy | Lower bound | Ratio |
|---|---:|---:|---:|
| Makespan | 408.5 h | 378.7 h | **1.08×** |
| Weighted tardiness | 2,716 | 529 | **11.8×** |

Makespan is nearly settled — at most 7% is schedulable away, and every policy
including the bad ones lands within 1.08–1.12×. Tardiness is not settled, but
the 11.8× is an *upper limit on remaining room*, not evidence that the room
exists: the LP relaxes preemption and every station but the binding one, so the
true optimum sits somewhere between the two.

---

## What went wrong, and what it taught

The corrected results are more useful than the headline ones.

### A reported 12.4% improvement was selection bias

Day 1 claimed a tuned weight vector beat the best classical rule by 12.4% on
weighted tardiness. It did not. The search took the best of 150 random
candidates scored on **five** seeds, then "validated" it on seeds 1–12 — a
superset containing those same five.

| Seed set | Improvement | Win rate | p |
|---|---:|---:|---:|
| Seeds 1–5 (the *selection* set) | +12.3% | 100% | 0.10 |
| Seeds 1–12 (contaminated) | +11.3% | 83% | 0.02 |
| **Seeds 2000–2039 (held out)** | **+0.4%** | 57% | **0.88** |

Taking the maximum of 150 noisy draws finds a lucky vector, not a good one.
Redone with disjoint selection and judging sets, the honest edge is +3.53 return
(75% wins, p=0.005). Seed ranges are now reserved by purpose and never reused.

### The reward was a myopia trap

Weighted tardiness is only booked when a job *finishes*, tens of decisions after
the dispatch that caused it — while the throughput term pays out immediately.
Measured at step 15 of an episode, the reward ranking of four dispatch rules was
the **exact reverse** of their full-episode ranking. LWKR looked best early and
finished worst.

PPO followed that gradient precisely, learning a direction between SPT and LWKR,
and lost to SPT significantly. Raising the discount to Monte Carlo returns made
it *worse* — with instance variance at ±38.7 against a policy effect near ±5,
the added variance costs more than the bias saves.

The fix was **potential-based reward shaping**: charge each job's projected
lateness as it accrues. The potential is zero on an empty floor — true at both
ends of an episode — so it telescopes away and total return is provably
unchanged (verified to 1e-14), while per-step credit assignment improves.

> PPO vs SPT: **−10.9% (p=0.013)** before shaping → **+0.3% (p=0.93)** after.

### The LP bound was invalid at first

The relaxation charged work at the *end* of its time slot, overstating lateness
by up to one slot per job. The tell: the "bound" *rose* from 450 to 632 as the
relaxation was **coarsened**, which is impossible for something valid. Charging
at the slot start fixed it; monotonicity under refinement is now a test.

### The line was too easy to schedule

The first configuration ran fine and produced clean KPIs — and all seven rules
landed within 4% of each other. A dispatch rule only decides anything when two
or more jobs are waiting, and Surface Treatment had a real choice **0%** of the
time. Across the line, 89% of dispatches were forced.

Fixed by matching capacity to work content, raising processing variability from
CV 0.15 to 0.50, and tightening due dates ~30% — deliberately *not* by
overloading the line, which gets contention at the cost of a factory that
physically cannot keep up. Bottlenecks now sit near 89%, and 34% of dispatches
are genuinely contested.

### Per-station weights did not help

The agent originally set one weight vector used by every station. That looked
like a real limitation: Surface Treatment has a single machine at ~89% load and
is the binding constraint, while Machining has three at ~72%, and a shared
vector cannot express "be different at the bottleneck".

It was tested and it does not help. Two independent lines of evidence:

| Test | Result |
|---|---|
| Fixed per-station search (24-dim, hill climbing) | +3.80 on selection seeds, **+0.16 held out** (p=0.88) |
| PPO with a 24-dim action | **−21.8% vs blend** (p=0.023) — worse than the shared version, which ties |

The search also converged to six near-identical variations on the same vector
rather than genuinely differentiated rules, which suggests there is not much
per-station structure to exploit here. The extra twenty dimensions bought
overfitting and a harder learning problem, not performance.

The capability is kept (`per_station=True` on the env, `TrainingConfig`) because
the negative result is worth having, and a 4-vector is still broadcast across
every station so the baselines run unchanged.

### Instance variance dominates

Across-seed standard deviation of episode return is **~38.7**. The seed-*paired*
difference between two policies is **~11.2**. Instance difficulty outweighs the
policy effect roughly 3.5 to 1, so two independently sampled means mostly
measure which job sets each policy happened to draw.

`evaluation.paired.compare()` therefore **raises** on mismatched seed lists
rather than returning a misleading number.

---

## Architecture

```
configs/factory.yaml          the whole line in one file
        │
src/dtmo/
├── digital_twin/             SimPy model
│   ├── entities.py           jobs, stations, families; slack and tardiness
│   ├── dispatch.py           the 4-weight composite rule
│   ├── stations.py           work centre; one process per machine
│   ├── factory.py            routing and the clock
│   └── kpis.py               the six KPIs
├── env/factory_env.py        Gymnasium env — 16-dim obs, 4-dim action
├── agents/                   policies and PPO training
├── evaluation/paired.py      seed-paired benchmarking
├── optimization/             lower bounds (combinatorial + LP via HiGHS)
├── visualization/            Plotly figures and the HTML dashboard
└── serving/                  FastAPI service
```

**Why stations do not use `simpy.Resource`.** `Resource` serves its queue FIFO,
and `PriorityResource` freezes priority at request time — but slack and waiting
time keep changing while a job sits in the queue, so a priority computed on
arrival is stale by the time a machine frees. Each station keeps an explicit
queue and each machine is its own process that re-scores at the instant it
becomes free.

**The environment steps the same simulation.** `FactoryModel.run()` (batch) and
`reset()`/`advance()` (stepped) produce **bit-identical** results — makespan
matches to four decimals. Pinned by `test_holding_weights_reproduces_the_batch_run`.

---

## Testing

```powershell
pytest                    # all 255
pytest -v                 # every name
pytest -k shaping         # by keyword
pytest --lf               # rerun last failures
```

| File | Tests | Guards |
|---|---:|---|
| `test_entities.py` | 17 | job/slack/tardiness arithmetic |
| `test_config.py` | 19 | YAML validation, load band |
| `test_dispatch.py` | 21 | the rule reproduces SPT/LPT/FIFO exactly |
| `test_factory.py` | 23 | simulation physics |
| `test_env.py` | 43 | Gym contract, action normalisation, shaping |
| `test_policies.py` | 19 | policy wrappers |
| `test_paired.py` | 22 | seed-paired statistics |
| `test_bounds.py` | 28 | lower-bound validity |
| `test_visualization.py` | 33 | chart rules, dashboard HTML |
| `test_serving.py` | 30 | API contract, train/serve skew |

The four that have each caught a real bug:

```powershell
pytest tests\test_factory.py::TestPhysics -v            # capacity, from the op log
pytest tests\test_env.py::TestPhysicsMatchesBatch -v    # stepping ≡ batch
pytest tests\test_bounds.py::TestLPValidity -v          # no policy beats the bound
pytest tests\test_serving.py::TestNoTrainServeSkew -v   # one observation encoder
```

`TestPhysics` reconstructs the schedule from each job's operation log and proves
no station ever ran more jobs at once than it has machines — it does not trust
the counters the simulator keeps.

---

## API

```powershell
python scripts\serve.py            # → http://127.0.0.1:8000/docs
python scripts\serve.py --demo     # exercise every endpoint in-process
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and which policy is loaded |
| `GET /info` | the configured line, and available policies |
| `POST /weights` | **the real one** — floor state in, four weights out |
| `POST /simulate` | run the twin end to end, with a lower bound |
| `GET /policies` | every policy the server can dispatch with |

The API speaks **floor state, not observation vectors**. A caller posts "9 jobs
queued at Surface Treatment, mean slack −6.5h" and gets back weights plus a
plain-language reading:

```
ppo    [+0.77 +0.37 +0.49 +0.19]  favours short jobs, then favours nearly-finished jobs
blend  [+0.89 +0.32 +0.16 -0.29]  favours short jobs, then favours urgent jobs
```

Encoding is the server's job, and it calls the same `encode_observation()` the
training environment does. A policy served a differently-scaled observation is
not degraded — it is reading noise, and every response would still look
well-formed.

### Docker

```powershell
docker build -t dtmo-serving:latest .          # ~4 min cold
docker run -d --name dtmo -p 8000:8000 dtmo-serving:latest
curl http://127.0.0.1:8000/health
docker stop dtmo && docker rm dtmo
```

Or with compose:

```powershell
docker compose up -d --build
docker compose logs -f
docker compose down
```

Built and verified: the image starts, the healthcheck reports `healthy`, the
trained policy loads from `/app/runs/ppo_shaped/ppo_best`, and both `/weights`
and `/simulate` return values identical to running locally.

The image is **2.05 GB**, of which the virtualenv is 1.49 GB. Torch installs
from PyTorch's CPU index deliberately — the default Linux wheel bundles CUDA and
would add roughly another gigabyte. Most of what remains is torch plus the
pandas/matplotlib that Stable-Baselines3 pulls in transitively. Serving a
16→64→64→4 MLP does not really need any of that: exporting the policy weights
and doing inference in numpy would drop torch and SB3 entirely and take the
image to a few hundred megabytes. Worth doing if image size ever matters.

---

## Configuration

Everything about the line lives in [`configs/factory.yaml`](configs/factory.yaml),
and the numbers there are tuned rather than guessed — the comments explain why.
`scripts/run_simulation.py` prints the analytic station load *before* simulating,
so a capacity problem is visible without blaming the scheduler.

The number to watch is **dispatch leverage**: the share of dispatches where two
or more jobs were actually waiting. Currently ~34%. If it collapses toward zero,
the KPIs stop responding to the weights and every downstream result becomes
meaningless — `test_queues_are_deep_enough_for_dispatch_to_matter` fails on
purpose to say so.

---

## Known limitations

- **PPO does not beat a well-tuned fixed rule.** It ties SPT and blend.
  Per-station weights were tried and made it worse. Whether SAC, a longer
  run, or a richer observation would close the gap is untested.
- **The tardiness bound is loose** (11.8× the best policy). It relaxes
  preemption and all but one station; an exact MILP would be needed to locate
  the true optimum.
- **Machine breakdowns, setup times, and batching are not modelled.** Heat
  Treatment in particular would be a batch oven in a real plant.

## Stack

SimPy · Gymnasium · Stable-Baselines3 · PyTorch (CPU) · SciPy/HiGHS · Plotly ·
FastAPI · pytest · Python 3.10
