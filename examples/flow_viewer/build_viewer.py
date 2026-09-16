"""Inline the traces, CSS and JS into one standalone HTML file.

python examples/export_traces.py && python examples/flow_viewer/build_viewer.py
open examples/flow_viewer/flow_viewer.html
"""

from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    # Reason: "</" inside inline JSON or JS would close the <script> tag early.
    data = (HERE / "traces.json").read_text().replace("</", "<\\/")
    html = (HERE / "viewer.html").read_text()
    for marker, content in (
        ("/*CSS*/", (HERE / "viewer.css").read_text()),
        ("/*DATA*/", data),
        ("/*VIEWS*/", (HERE / "views.js").read_text()),
        ("/*APP*/", (HERE / "viewer.js").read_text()),
    ):
        assert html.count(marker) == 1, marker
        html = html.replace(marker, content)
    out = HERE / "flow_viewer.html"
    out.write_text(html)
    print(f"{out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
