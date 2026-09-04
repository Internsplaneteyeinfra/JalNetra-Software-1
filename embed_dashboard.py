"""Embed dashboard_data.json into dashboard.html for standalone use."""
import pathlib

root = pathlib.Path(__file__).parent / "dashboard" / "mula_mutha"
html = (root / "dashboard.html").read_text(encoding="utf-8")
data = (root / "dashboard_data.json").read_text(encoding="utf-8")
marker = '<script id="fallback" type="application/json">'
end = "</script>"
i = html.find(marker)
if i < 0:
    raise SystemExit("fallback marker not found")
j = html.find(end, i)
html = html[: i + len(marker)] + data + html[j:]
out = pathlib.Path(r"c:\Users\Sanskruti.Jadhav\Desktop\sanskruti\jalnetra_mula_mutha_dashboard.html")
out.write_text(html, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
