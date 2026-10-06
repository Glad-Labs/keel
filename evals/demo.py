"""Build a demo journal from the fictional one in evals/fixture.

    .venv/bin/python -m evals.demo .demo-home
    .venv/bin/keel --home .demo-home serve

Handy for trying Keel (or showing it to someone) without touching your own
journal. Everything in it is made up.
"""

import sys
from pathlib import Path

from evals.run import build_template
from keel import config

if __name__ == "__main__":
    home = Path(sys.argv[1] if len(sys.argv) > 1 else ".demo-home")
    if home.exists():
        raise SystemExit(f"{home} already exists; delete it first to rebuild")
    build_template(home, config.DEFAULTS["embed_model"])
    print(f"Demo journal ready in {home}. Try: keel --home {home} serve")
