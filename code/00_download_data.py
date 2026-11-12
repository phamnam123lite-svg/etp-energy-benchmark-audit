"""OPTIONAL - re-download the raw files from their public sources and compare SHA-256 checksums.

NOT EXECUTED BY THE AUTHOR: the authoring environment had no direct network access (the files in
data/raw were fetched through a web browser, see data/README.md). The script uses only the Python
standard library and writes to data/raw_redownload/, never over data/raw/.

A checksum mismatch for a Melbourne Water table means the publisher has revised the table since
12 November 2026 (or that this script serialises a value differently from the original retrieval);
it does not by itself indicate a problem with the stored copy.

Usage:  python 00_download_data.py
"""
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from common import ROOT, RAW_SHA256

OUT = ROOT / "data" / "raw_redownload"
ARCGIS = "https://services5.arcgis.com/ZSYwjtv8RKVhkXIL/arcgis/rest/services/{}/FeatureServer/0"
MENDELEY_FILES = "https://data.mendeley.com/public-api/datasets/pprkvz3vbd/files?folder_id=root&version=1"
DAILY = ["MWC_ETP_Daily_ElectricityUse_From2014", "MWC_ETP_Daily_InfluentQuality_From2014", "MWC_ETP_Daily_EffluentQuality_From2014"]
HOURLY = ["MWC_ETP_Hourly_InfluentFlow_From2009", "MWC_ETP_Hourly_EffluentFlow_From2009"]


def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "etp-audit/0.2", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def js_str(v):
    """Text form of a JSON value as produced by JavaScript's String(): 3.0 -> '3', None -> ''."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) < 1e21 else repr(v)
    return str(v).replace(",", ";")


def fetch_table(service):
    base = ARCGIS.format(service)
    fields = [f["name"] for f in json.loads(get(base + "?f=json"))["fields"]]
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "orderByFields": "ObjectId ASC", "resultOffset": off,
                                    "resultRecordCount": 2000, "returnGeometry": "false", "f": "json"})
        j = json.loads(get(base + "/query?" + q))
        if "error" in j:
            raise RuntimeError(j["error"])
        feats = j.get("features", [])
        rows += [[f["attributes"].get(k) for k in fields] for f in feats]
        off += len(feats)
        if not feats or not j.get("exceededTransferLimit"):
            break
    return fields, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    built = {}
    info = json.loads(get(MENDELEY_FILES, {"Accept": "application/vnd.mendeley-public-dataset.1+json"}))[0]
    built[info["filename"]] = get(info["content_details"]["download_url"])
    for s in DAILY:
        fields, rows = fetch_table(s)
        text = ",".join(fields) + "\n" + "\n".join(",".join(js_str(v) for v in r) for r in rows) + "\n"
        built[s + ".csv"] = text.encode("utf-8")
    for s in HOURLY:
        fields, rows = fetch_table(s)
        i_s, i_v = fields.index("recordstart"), next(i for i, f in enumerate(fields) if f.startswith("ETP_"))
        lines = ["start_epoch_hour," + fields[i_v]] + [js_str(r[i_s] / 3600000) + "," + js_str(r[i_v]) for r in rows]
        built[s + "_compact.csv"] = ("\n".join(lines) + "\n").encode("utf-8")
    for name, data in built.items():
        (OUT / name).write_bytes(data)
        h = hashlib.sha256(data).hexdigest()
        ref = RAW_SHA256.get(name)
        print(f"{name:55s} {len(data):9d} bytes  {'MATCH' if h == ref else 'DIFFERS from stored copy'}")


if __name__ == "__main__":
    main()
