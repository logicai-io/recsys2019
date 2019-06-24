import subprocess
import os

ps = []
for n in range(32):
    args = ["python", "generate_training_data.py", "--limit", "2000000", "--hashn", str(n)]
    p = subprocess.Popen(args)
    ps.append(p)

for p in ps:
    p.wait()

os.system("paste -d, ../../../data/events_sorted_trans_0*.csv > ../../../data/events_sorted_trans_all.csv")
