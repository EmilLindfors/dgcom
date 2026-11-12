# /// script
# requires-python = ">=3.10"
# ///
"""An IOC Sea Level Station Monitoring Facility record (JSON) as the viewer's gauge
file (`scripts/kartverket_gauge.sh` format), about the record's own mean.
usage: uv run scripts/ioc_gauge_file.py record.json name code lat lon out.txt
(the records are the JSON `scripts/ioc_gauge_skill.py` keeps under data/tide_gauges/ioc/)"""
import json, sys
from collections import Counter

rec, name, code, lat, lon, out = sys.argv[1:7]
r = json.load(open(rec))
sensor = Counter(x["sensor"] for x in r).most_common(1)[0][0]
rows = [(x["stime"].strip(), float(x["slevel"])) for x in r if x["sensor"] == sensor]
mean = sum(v for _, v in rows) / len(rows)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(f"# IOC Sea Level Station Monitoring Facility (SHOA), sensor {sensor}, {rows[0][0]} to {rows[-1][0]}\n")
    f.write(f"# source: https://www.ioc-sealevelmonitoring.org/station.php?code={code}\n")
    f.write(f"# station: {name}\n# id: {code}\n# latitude: {lat}\n# longitude: {lon}\n")
    f.write("# datum: the record's mean\n# units: m\n# columns: datetime water_level(m)\n")
    for t, v in rows:
        f.write(f"{t.replace(' ', 'T')}Z {v - mean:.3f}\n")
print(len(rows), "rows, mean removed", round(mean, 3))
