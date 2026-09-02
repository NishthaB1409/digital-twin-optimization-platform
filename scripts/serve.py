"""Day 6: run the scheduling API locally.

    python scripts/serve.py
    python scripts/serve.py --model runs/ppo_shaped/ppo_best --reload
    python scripts/serve.py --demo          # no server; exercise it in-process

Once running, http://127.0.0.1:8000/live watches the twin run on a wall clock
with the policy steering it -- the same control loop a real deployment uses,
with a simulated floor standing in for the physical one.

Once running, the interactive docs are at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import json
import os
import warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default="runs/ppo_shaped/ppo_best",
        help="trained agent; falls back to the tuned fixed rule if absent",
    )
    parser.add_argument("--config", default=None, help="factory YAML")
    parser.add_argument("--reload", action="store_true", help="restart on edits")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="call every endpoint in-process and print the results",
    )
    return parser


def demo() -> None:
    """Exercise the API without binding a port."""
    from fastapi.testclient import TestClient

    from dtmo.serving.app import app
    from dtmo.utils.config import load_config

    client = TestClient(app)
    config = load_config(os.environ.get("DTMO_CONFIG") or None)
    capacities = {s.name: s.capacity for s in config.stations}

    print("GET /health")
    print(" ", json.dumps(client.get("/health").json()))

    info = client.get("/info").json()
    print("\nGET /info")
    print(f"  line     : {', '.join(info['stations'])}")
    print(f"  policy   : {info['policy_loaded']}  ({info['model_path']})")

    # A congested floor: the bottleneck stations are backed up and the average
    # job has already lost its slack.
    floor = {
        "stations": [
            {
                "name": name,
                "queue_length": 9 if capacities[name] == 1 else 3,
                "busy_machines": capacities[name],
            }
            for name in config.station_names
        ],
        "clock_hours": 180.0,
        "jobs_completed": 40,
        "jobs_in_progress": 22,
        "total_jobs": config.n_jobs,
        "mean_slack_hours": -6.5,
    }

    print("\nPOST /weights   (bottlenecks backed up, average job already late)")
    for policy in ("ppo", "blend", "spt"):
        body = client.post(f"/weights?policy={policy}", json=floor).json()
        w = body["weights"]
        print(
            f"  {policy:<6} [{w['processing_time']:+.2f} {w['slack']:+.2f} "
            f"{w['remaining_work']:+.2f} {w['waiting_time']:+.2f}]  {body['rule']}"
        )

    print("\nPOST /simulate")
    for policy in ("spt", "mwkr"):
        body = client.post(
            "/simulate", json={"policy": policy, "seed": 1000, "n_jobs": 60}
        ).json()
        print(
            f"  {policy:<6} makespan {body['makespan_hours']:7.1f}h  "
            f"on-time {body['on_time_rate']:5.1%}  "
            f"wgt tardiness {body['total_weighted_tardiness']:8.1f}  "
            f"(bound {body['weighted_tardiness_lower_bound']:.1f})"
        )

    print("\nendpoints verified in-process; run without --demo to bind a port")


def main() -> None:
    warnings.simplefilter("ignore")
    args = build_parser().parse_args()

    os.environ["DTMO_MODEL"] = args.model
    if args.config:
        os.environ["DTMO_CONFIG"] = args.config

    if args.demo:
        demo()
        return

    import uvicorn

    print(f"  API docs   http://{args.host}:{args.port}/docs")
    print(f"  LIVE FLOOR http://{args.host}:{args.port}/live")
    uvicorn.run(
        "dtmo.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
