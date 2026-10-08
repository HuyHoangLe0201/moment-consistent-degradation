"""Reproduce every number, table and figure of the paper, in order.

    python fetch_data.py         # once: the published data sets
    python run_all.py            # everything (about an hour on a laptop)
    python run_all.py --quick    # tables and figures from the stored results
    python run_all.py --from study_realdata    # resume from one step

Each step writes JSON into results/; make_tables.py turns the JSON into the
LaTeX tables in results/tables/, and figures/export_figures.py writes the
TikZ figures into figures/out/.  All simulations are seeded.
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")

# (script, what it produces, paper location)
STEPS = [
    ("study_gridconsistency", "grid study", "Sec. 5.1, Fig. 5, Table D.1"),
    ("study_lamperti", "Lamperti reduction", "Sec. 5.3, Fig. 3, Table D.3"),
    ("study_igapprox", "IG approximation", "Sec. 5.3, Table D.4"),
    ("study_singlerate", "one-record identifiability", "Sec. 5.2, Table D.2"),
    ("study_weibull_check", "Weibull steps", "Sec. 5.3, Table 3"),
    ("make_maintenance_example", "replacement risk", "Sec. 7.1, Fig. 7"),
    ("study_realdata", "variance growth, likelihood", "Sec. 6, Table 4, Fig. 6"),
    ("study_review", "sum laws, families", "Table 2, Table 5"),
    ("study_alloy_clock", "Alloy-A on the damage clock", "Sec. 6, Table 4"),
    ("study_application", "lasers end to end", "Sec. 7.2, Table 6, Fig. 8"),
    ("study_application_deviceb", "Device-B end to end", "Sec. 7.2, Table 6, Fig. 8"),
    ("study_application_alloy", "Alloy-A end to end", "Sec. 7.2, Table 6, Fig. 8"),
    ("study_brier", "Brier score of the event forecasts", "Sec. 7.2, Table D.7"),
    ("study_bounds", "explicit error bounds", "App. B, Table B.1"),
    ("study_sensitivity", "coupling-parameter sensitivity", "Sec. 5.3, Table D.6"),
]


def run(path):
    t0 = time.time()
    sys.path.insert(0, os.path.dirname(path))
    runpy.run_path(path, run_name="__main__")
    print(f"     ({time.time() - t0:.0f} s)")


def main():
    quick = "--quick" in sys.argv
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    os.chdir(HERE)
    for sub in ("theory", "realdata", "tables", "figures"):
        os.makedirs(os.path.join(HERE, "results", sub), exist_ok=True)
    sys.path.insert(0, CODE)
    if not quick:
        names = [s[0] for s in STEPS]
        first = names.index(start) if start else 0
        for name, what, where in STEPS[first:]:
            print(f"== {name}: {what}  [{where}]")
            run(os.path.join(CODE, name + ".py"))
    print("== tables")
    run(os.path.join(CODE, "make_tables.py"))
    print("== figures")
    run(os.path.join(HERE, "figures", "export_figures.py"))


if __name__ == "__main__":
    main()
