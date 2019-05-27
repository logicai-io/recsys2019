import csv

reader = csv.DictReader(open("../data/events_sorted_trans.csv"))

for i, row in enumerate(reader):
    if i % 10000 == 0:
        print(i)
