import subprocess
import os

#ps = []
#for n in range(32):
#    args = ["python", "generate_training_data.py", "--hashn", str(n)]
#    p = subprocess.Popen(args)
#    ps.append(p)

#for p in ps:
#    p.wait()

os.system(
    "paste -d, ../../../data/events_sorted_trans_0*.csv /sdb/price_features.csv /sdb/comp-v1-selected.csv /sdb/time_rank_user-features.csv > /sdb/all.csv"
)
