#!/usr/bin/env python3
"""
compare_dual_replica.py

Join the two per-replica CSVs produced by dual_replica_sync_driver.sh (via
benchmark_parser.py) against a single-replica 1P1D baseline (e.g. job 200584)
and answer the scaling question directly, per (ISL, OSL, Concurrency) cell:

  * A_total / B_total   -- each replica's Total_Token_Throughput_tok_s
  * dual_total          -- A_total + B_total (aggregate of the 4-node deployment)
  * baseline_total      -- the single-replica 1P1D value at the SAME cell
  * scaling_vs_single   -- dual_total / baseline_total  (2.0 == perfect doubling)
  * replicaA_vs_base    -- A_total / baseline_total  (~1.0 == a replica matches
                           the standalone box, i.e. no cross-replica interference)
  * balance_pct         -- |A-B| / max(A,B) * 100   (0 == perfectly balanced load)

Also emits the same joins for Output_Token_Throughput_tok_s and reports median
TTFT so you can confirm latency didn't move when you doubled the hardware.

Pure stdlib (csv only) so it runs on the login node without pandas.

Usage:
  compare_dual_replica.py --a replicaA.csv --b replicaB.csv \
      [--baseline 200584_1p1d.csv] -o comparison.csv
"""
import argparse
import csv
import sys
from collections import OrderedDict

KEY = ("ISL", "OSL", "Concurrency")


def _f(row, col):
    v = row.get(col, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load(path):
    """Return {(isl,osl,con): row} keyed by the measurement cell."""
    out = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                k = (int(float(row["ISL"])), int(float(row["OSL"])),
                     int(float(row["Concurrency"])))
            except (KeyError, ValueError):
                continue
            out[k] = row
    return out


def pct(a, b):
    hi = max(a, b)
    return 0.0 if hi == 0 else abs(a - b) / hi * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="replica A CSV")
    ap.add_argument("--b", required=True, help="replica B CSV")
    ap.add_argument("--baseline", help="single-replica 1P1D baseline CSV (e.g. job 200584)")
    ap.add_argument("-o", "--output", default="comparison.csv", help="output CSV path")
    args = ap.parse_args()

    A = load(args.a)
    B = load(args.b)
    base = load(args.baseline) if args.baseline else {}

    cells = [k for k in A if k in B]
    if not cells:
        print("ERROR: no overlapping (ISL,OSL,Concurrency) cells between A and B.",
              file=sys.stderr)
        sys.exit(1)
    cells.sort(key=lambda k: (k[0], k[1], k[2]))

    cols = [
        "ISL", "OSL", "Concurrency",
        "A_total_tok_s", "B_total_tok_s", "dual_total_tok_s",
        "baseline_total_tok_s", "scaling_vs_single", "replicaA_vs_base",
        "balance_pct",
        "A_out_tok_s", "B_out_tok_s", "dual_out_tok_s",
        "A_median_ttft_ms", "B_median_ttft_ms", "baseline_median_ttft_ms",
        "A_failed", "B_failed",
    ]

    rows = []
    for k in cells:
        a, b = A[k], B[k]
        at, bt = _f(a, "Total_Token_Throughput_tok_s"), _f(b, "Total_Token_Throughput_tok_s")
        ao, bo = _f(a, "Output_Token_Throughput_tok_s"), _f(b, "Output_Token_Throughput_tok_s")
        dual = (at or 0) + (bt or 0)
        dual_out = (ao or 0) + (bo or 0)
        bt_base = _f(base[k], "Total_Token_Throughput_tok_s") if k in base else None
        scaling = round(dual / bt_base, 3) if bt_base else ""
        a_vs_base = round((at or 0) / bt_base, 3) if bt_base else ""
        rows.append({
            "ISL": k[0], "OSL": k[1], "Concurrency": k[2],
            "A_total_tok_s": at, "B_total_tok_s": bt,
            "dual_total_tok_s": round(dual, 2),
            "baseline_total_tok_s": bt_base if bt_base is not None else "",
            "scaling_vs_single": scaling,
            "replicaA_vs_base": a_vs_base,
            "balance_pct": round(pct(at or 0, bt or 0), 2),
            "A_out_tok_s": ao, "B_out_tok_s": bo, "dual_out_tok_s": round(dual_out, 2),
            "A_median_ttft_ms": _f(a, "Median_TTFT_ms"),
            "B_median_ttft_ms": _f(b, "Median_TTFT_ms"),
            "baseline_median_ttft_ms": (_f(base[k], "Median_TTFT_ms") if k in base else ""),
            "A_failed": a.get("Failed", ""), "B_failed": b.get("Failed", ""),
        })

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Console summary.
    print(f"Wrote {len(rows)} cells -> {args.output}\n")
    hdr = f"{'ISL':>6} {'OSL':>6} {'con':>5} {'A_tot':>9} {'B_tot':>9} {'dual':>10}"
    if base:
        hdr += f" {'base':>9} {'x_single':>9}"
    hdr += f" {'bal%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['ISL']:>6} {r['OSL']:>6} {r['Concurrency']:>5} "
                f"{(r['A_total_tok_s'] or 0):>9.1f} {(r['B_total_tok_s'] or 0):>9.1f} "
                f"{r['dual_total_tok_s']:>10.1f}")
        if base:
            bt = r["baseline_total_tok_s"]
            line += f" {(bt if bt != '' else 0):>9.1f} {str(r['scaling_vs_single']):>9}"
        line += f" {r['balance_pct']:>6.1f}"
        print(line)

    if base:
        vals = [r["scaling_vs_single"] for r in rows if r["scaling_vs_single"] != ""]
        if vals:
            print(f"\nMean scaling vs single replica: {sum(vals)/len(vals):.3f}x "
                  f"(2.000x == perfect doubling)")


if __name__ == "__main__":
    main()
