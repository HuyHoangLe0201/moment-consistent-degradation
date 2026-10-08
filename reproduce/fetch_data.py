"""Download the published degradation data sets used in Section 6 and 7.2.

They come from the R package that accompanies Meeker & Escobar, Statistical
Methods for Reliability Data (SMRD, github.com/Auburngrads/SMRD).  They are not
redistributed with this repository; this script fetches them into
data_external/smrd/.

    python fetch_data.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data_external", "smrd")
BASE = "https://raw.githubusercontent.com/Auburngrads/SMRD/master/data/"
FILES = ("gaaslaser", "deviceb", "alloya", "metalwear", "bkfatigue10",
         "pipelinethickness")


def main():
    os.makedirs(DEST, exist_ok=True)
    for name in FILES:
        path = os.path.join(DEST, name + ".RData")
        if os.path.exists(path):
            print(f"  have {name}.RData")
            continue
        urllib.request.urlretrieve(BASE + name + ".RData", path)
        print(f"  got  {name}.RData ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
