import pathlib
import warnings

warnings.filterwarnings("ignore")
SORTED_TRANS_CSV = (
    pathlib.Path().absolute().parents[1] / "data" / "events_sorted_trans.csv"
)
print(pathlib.Path().absolute())
print(SORTED_TRANS_CSV)
