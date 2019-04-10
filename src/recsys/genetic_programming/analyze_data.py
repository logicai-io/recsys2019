import json

with open("click_indices.ndjson") as inp:
    lines = inp.readlines()

lines = list(map(json.loads, lines))
