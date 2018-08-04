#!/usr/bin/env python3

import csv, json
from util import *

HEADERS = 3

data = []

with open("data.csv") as f:
    lines = csv.reader(f, delimiter=',')
    headers = None
    for l, line in enumerate(lines):
        if headers is None:
            headers = [None] * len(line)
        if l < HEADERS:
            for h, header in enumerate(line):
                # if headers[h] is None and len(header.strip()) > 1:
                if len(header.strip()) > 1:                    
                    if header in headers:
                        header = "%s_" % header
                    headers[h] = header
        elif line[0].lower().strip() == "average":
            continue
        else:
            line = [as_numeric(item) if len(item) else None for item in line]
            data.append(dict(zip(headers, line)))
            print(json.dumps(data, indent=4))

save("data.pkl", data)