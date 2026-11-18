#!/usr/bin/env python
"""results/v2/report.html -> results/v2/report.pdf without a browser.

The page's script is executed in QuickJS against a tiny DOM stub, so every chart and table is baked in as static
SVG/HTML; equations are pre-rendered with KaTeX; WeasyPrint lays out the A4 PDF. Optional dependencies:

    pip install quickjs weasyprint brotli
    python scripts/report_pdf.py                      # after `pgm v2 report`

KaTeX (script, stylesheet, fonts) is downloaded once into --assets (default ~/.cache/pgm-report/katex-0.16.9).
The PDF shows one view of the interactive parts (curve explorer: Llama-3.1-8B MATH500, cross-fitted points).
"""
from __future__ import annotations

import argparse
import html as H
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/"
LIGHT = {"--bg": "#f7f8fa", "--surface": "#fcfcfb", "--ink": "#12151b", "--ink-2": "#4c5260", "--ink-3": "#7a8090", "--rule": "#dde1e8",
         "--accent": "#1c5cab", "--soft": "#eaf1fb", "--s1": "#2a78d6", "--s2": "#eb6834", "--s3": "#1baf7a", "--s4": "#eda100",
         "--s5": "#e87ba4", "--s6": "#008300", "--s7": "#6250d6", "--s8": "#e34948", "--sc": "#8a8985", "--good": "#0ca30c", "--bad": "#d03b3b"}
# WeasyPrint's SVG renderer ignores document CSS classes: these are inlined.
SVG_STYLE = {
    "dg-node": "fill:#fcfcfb;stroke:#12151b;stroke-width:1.6", "dg-node obs": "fill:#eaf1fb;stroke:#1c5cab;stroke-width:1.6",
    "dg-node det": "fill:#fcfcfb;stroke:#12151b;stroke-width:1.6", "dg-node off": "fill:none;stroke:#7a8090;stroke-width:1.6;stroke-dasharray:4 3",
    "dg-hyper": "fill:#12151b", "dg-m": "fill:#12151b;font-style:italic;font-size:18px;font-family:KaTeX_Main,serif",
    "dg-m off": "fill:#7a8090;font-style:italic;font-size:18px;font-family:KaTeX_Main,serif",
    "dg-s": "fill:#4c5260;font-size:12px;font-family:IBM Plex Sans KR", "dg-s off": "fill:#7a8090;font-size:12px;font-family:IBM Plex Sans KR",
    "dg-edge": "fill:none;stroke:#4c5260;stroke-width:1.4", "dg-edge off": "fill:none;stroke:#7a8090;stroke-width:1.4;stroke-dasharray:4 3",
    "dg-plate": "fill:none;stroke:#dde1e8;stroke-width:1.5", "dg-pl": "fill:#7a8090;font-size:12px;font-family:IBM Plex Mono",
    "axis": "fill:#7a8090;font-size:11px;font-family:IBM Plex Mono", "grid": "stroke:#dde1e8;stroke-width:1",
    "axlabel": "fill:#4c5260;font-size:12px;font-family:IBM Plex Sans KR", "rowlabel": "fill:#4c5260;font-size:12px;font-family:IBM Plex Sans KR",
}
DOM_STUB = r"""
var _ids = {};
function El(id){ this.id=id; this._html=''; this.style={}; this.value='0'; this.dataset={}; this._q={}; }
El.prototype = {
  set innerHTML(v){ this._html=String(v); }, get innerHTML(){ return this._html; },
  querySelector(sel){ if(!this._q[sel]) this._q[sel]=new El(this.id+' '+sel); return this._q[sel]; },
  querySelectorAll(){ return []; }, addEventListener(){}, setAttribute(){},
  getBoundingClientRect(){ return {left:0, top:0, width:900, height:420}; }
};
var document = { getElementById(id){ if(!_ids[id]) _ids[id]=new El(id); return _ids[id]; } };
"""
PRINT_CSS = """
@page { size: A4; margin: 13mm 11mm 15mm; @bottom-center { content: counter(page) " / " counter(pages); font: 9pt sans-serif; color: #7a8090; } }
body { background: #fff; } .wrap { max-width: none; padding: 0; } nav.toc, .tip, .controls { display: none; }
.chart, .eq, .diagram, .vcard, .multi > div, tr { break-inside: avoid; } h2, h3, .secno { break-after: avoid; }
.tbl, .eq { overflow: visible; } .katex { font-size: 1.08em; } .diagram svg { min-width: 0; }
.verdict { display: block; } .vcard { display: inline-block; width: 48.5%; vertical-align: top; margin: 0 1% 10px 0; }
.legend { display: block; } .legend span { display: inline-block; margin: 0 16px 4px 0; white-space: nowrap; }
.legend svg, .chart .legend svg { vertical-align: middle; margin-right: 4px; width: auto; height: auto; display: inline-block; }
table { font-size: 11px; } th, td { padding: 4px 5px; } th { font-size: 10.5px; } td.n, th.n { white-space: normal; }
.multi { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
"""


def katex_assets(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["katex.min.js", "katex.min.css"]:
        if not (folder / name).exists():
            urllib.request.urlretrieve(KATEX + name, folder / name)
    for font in sorted(set(re.findall(r"fonts/KaTeX_[\w-]+\.woff2", (folder / "katex.min.css").read_text()))):
        target = folder / font
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(KATEX + font, target)
    return folder


def static_html(page: str, assets: Path) -> str:
    import quickjs
    script = re.findall(r"<script>(.*?)</script>", page, re.S)[-1]
    ctx = quickjs.Context()
    ctx.eval(DOM_STUB + script)
    filled = json.loads(ctx.eval("JSON.stringify(Object.fromEntries(Object.entries(_ids).map(([k, v]) => [k, v._html])))"))
    shown = ctx.eval("C[ex.cell].model + ' · ' + C[ex.cell].dataset")
    for key, value in filled.items():
        if key == "ex-cell":
            page = page.replace('<select id="ex-cell"></select>', f"<b>{H.escape(shown)}</b>")
            continue
        page = re.sub(r'(<div[^>]*\bid="%s"[^>]*>)(</div>)' % re.escape(key), lambda m: m.group(1) + value + m.group(2), page, count=1)
    page = page.replace('<div class="chart"><div class="legend" id="ex-leg">',
                        f'<p class="note">표시: <b>{H.escape(shown)}</b> · 비교 묶음: baseline · 운영점: cross-fitting</p>'
                        '<div class="chart"><div class="legend" id="ex-leg">', 1)
    page = page.replace('<div class="chart"><div class="legend" id="mt-leg">',
                        '<p class="note">표시: cross-fitting 운영점 · baseline은 같은 규칙으로 튜닝 (웹 버전에서 전환 가능)</p>'
                        '<div class="chart"><div class="legend" id="mt-leg">', 1)
    kctx = quickjs.Context()
    kctx.eval("var window=this; var self=this; var module=undefined; var exports=undefined; var define=undefined;")
    kctx.eval((assets / "katex.min.js").read_text(encoding="utf-8"))

    def tex(match, display):
        source = json.dumps(H.unescape(match.group(1)))
        return kctx.eval(f"katex.renderToString({source},{{displayMode:{'true' if display else 'false'},output:'html',throwOnError:true}})")
    head, body = page.split("<body>", 1)
    body = re.sub(r"<script>.*?</script>", "", body, flags=re.S)
    body = re.sub(r"\\\[(.*?)\\\]", lambda m: tex(m, True), body, flags=re.S)
    body = re.sub(r"\\\((.*?)\\\)", lambda m: tex(m, False), body, flags=re.S)
    head = re.sub(r"<script[^>]*>.*?</script>", "", head, flags=re.S)
    page = head + "<body>" + body
    page = re.sub(r"<svg.*?</svg>", lambda m: re.sub(
        r'class="([^"]+)"', lambda c: c.group(0) + (f' style="{SVG_STYLE[c.group(1)]}"' if c.group(1) in SVG_STYLE else ""), m.group(0)),
        page, flags=re.S)
    for var, value in LIGHT.items():
        page = page.replace(f"var({var})", value)
    return page.replace("</head>", f'<link rel="stylesheet" href="{(assets / "katex.min.css").as_uri()}"><style>{PRINT_CSS}</style></head>', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", default=str(ROOT / "results/v2/report.html"))
    parser.add_argument("--out", default=str(ROOT / "results/v2/report.pdf"))
    parser.add_argument("--assets", default=str(Path.home() / ".cache/pgm-report/katex-0.16.9"))
    parser.add_argument("--zoom", type=float, default=0.58, help="CSS px to page scale (0.58 keeps the desktop layout on A4)")
    a = parser.parse_args()
    from weasyprint import HTML
    page = static_html(Path(a.html).read_text(encoding="utf-8"), katex_assets(Path(a.assets)))
    HTML(string=page, base_url=a.html).write_pdf(a.out, zoom=a.zoom)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
