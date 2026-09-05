"""Export the trained policy to numpy, so serving needs no PyTorch.

    python scripts/export_policy.py
    python scripts/export_policy.py --model runs/ppo_perstation/ppo_best

The export is checked against the torch model before it is written. If they
disagree by more than a rounding error the export fails rather than shipping a
policy that quietly behaves differently.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path


def main() -> None:
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/ppo_shaped/ppo_best")
    parser.add_argument("--out", default="runs/ppo_shaped/policy.npz")
    parser.add_argument("--checks", type=int, default=512)
    args = parser.parse_args()

    from dtmo.serving.export import NumpyPolicy, export_policy

    path, worst = export_policy(args.model, args.out, n_checks=args.checks)
    policy = NumpyPolicy.load(path)

    source = Path(f"{args.model}.zip")
    if not source.exists():
        source = Path(args.model)

    print(f"exported  {args.model}  ->  {path}")
    print(f"agreement worst difference {worst:.2e} over {args.checks} observations")
    print(f"network   {policy!r}  ({policy.n_parameters:,} parameters)")
    print(
        f"size      {source.stat().st_size / 1024:.0f} KB  ->  "
        f"{path.stat().st_size / 1024:.0f} KB"
    )
    print("\nserve it with:")
    print(f"  DTMO_POLICY={path} python scripts/serve.py")
    print("  docker build -f Dockerfile.serve -t dtmo-serve:slim .")


if __name__ == "__main__":
    main()
