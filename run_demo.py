"""Run the whole mock demo, then open the flow viewer.

    python run_demo.py            # tests, console demo, traces, viewer, then opens it in the browser
    python run_demo.py --no-open  # same, without opening the browser

No API keys needed: tool results and model answers are scripted, everything
else (graph, rules, prompts, Salesforce mapping) is the real code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
EXAMPLES = ROOT / "examples"
VIEWER = EXAMPLES / "flow_viewer" / "flow_viewer.html"


def step(title: str, args: list[str]) -> None:
    print(f"\n==> {title}\n    $ {' '.join(args)}", flush=True)
    # Reason: the example scripts import their neighbors (mock_data, scenarios),
    # so they run with examples/ on the path, the same as `python examples/<script>.py`.
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStopped: '{title}' failed with exit code {result.returncode}. See the output above.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-open", action="store_true", help="do not open the viewer in a browser")
    parser.add_argument("--skip-tests", action="store_true", help="skip the unit tests")
    opts = parser.parse_args()

    if not opts.skip_tests:
        step("1/4 Unit tests", ["-m", "pytest", "-q"])
    step("2/4 Console demo: one run, one SDR rerun, one Salesforce sync", [str(EXAMPLES / "run_mock_demo.py")])
    step("3/4 Five scenarios through the real graph, recorded step by step", [str(EXAMPLES / "export_traces.py")])
    step("4/4 Build the flow viewer", [str(EXAMPLES / "flow_viewer" / "build_viewer.py")])

    print(f"\nFlow viewer: {VIEWER}")
    if not opts.no_open:
        webbrowser.open(VIEWER.resolve().as_uri())


if __name__ == "__main__":
    main()
