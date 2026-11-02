import json, os
H = os.path.dirname(__file__)
read = lambda n: open(os.path.join(H, n), encoding="utf-8").read()
data = open(os.path.join(H, "data", "report_data.json")).read()
js = read("page_js.html").replace("/*DATA*/", data)
html = read("page_head.html") + read("page_body_a.html") + read("page_body_b.html") + read("page_body_c.html") + js
open(os.path.join(H, "spade_report.html"), "w", encoding="utf-8").write(html); print(len(html))
