from csv import DictReader
from tqdm import tqdm

with open("../../data/train.csv") as inp:
    dr = DictReader(inp)
    for row in tqdm(dr):
        pass
