from functools import reduce
from multiprocessing.pool import Pool

import click
from joblib import Parallel, delayed
from recsys.data_generator.accumulators import get_accumulators, logger, get_accumulators_basic
from recsys.data_generator.data_io_chunks import DataIOChunks
from tqdm import tqdm


def merge_dicts(*dicts):
    return reduce(lambda d1, d2: reduce(lambda d, t:
                                        dict(list(d.items()) + [t]),
                                        d2.items(), d1),
                  dicts, {})


class FeatureGenerator:
    def __init__(self, data_io: DataIOChunks, limit, accumulators, accs_by_action_type):
        self.data_io = data_io
        self.limit = limit
        self.accumulators = accumulators
        self.accs_by_action_type = accs_by_action_type
        self.pool = Pool(8)
        print("Number of accumulators %d" % len(self.accumulators))

    def calculate_features_per_item(self, acc, clickout_id, item_id, price, rank, row):
        obs = row.copy()
        obs["item_id"] = item_id
        obs["item_id_clicked"] = row["reference"]
        obs["was_clicked"] = int(row["reference"] == item_id)
        obs["clickout_id"] = clickout_id
        obs["rank"] = rank
        obs["price"] = price
        return self.generate_features_from_acc(acc, obs, row)

    def generate_features_from_acc(self, acc, obs, row):
        new_obs = {}
        value = acc.get_stats(row, obs)
        if isinstance(value, dict):
            for k, v in value.items():
                new_obs[k] = v
        else:
            new_obs[acc.name] = acc.get_stats(row, obs)
        return new_obs

    def generate_features(self):
        logger.info("Starting feature generation")
        logger.info("Starting processing")
        self.data_io.process(self.process_chunk)

    def process_chunk(self, rows):
        rows_prepared = [self.prepare_row(row) for row in rows]
        outputs = []
        outputs.append(self.process_rows(acc=None, rows=rows_prepared))
        outputs_par = Parallel(n_jobs=8)(delayed(acc)(rows_prepared) for acc in self.accumulators)
        outputs.extend(outputs_par)
        observations = [merge_dicts(*out) for out in zip(*outputs)]
        return observations

    def prepare_row(self, row):
        row = row.copy()
        row["timestamp"] = int(row["timestamp"])
        row["fake_impressions_raw"] = row["fake_impressions"]
        row["fake_impressions"] = row["fake_impressions"].split("|")
        row["fake_index_interacted"] = (
            row["fake_impressions"].index(row["reference"])
            if row["reference"] in row["fake_impressions"]
            else -1000
        )
        if row["action_type"] == "clickout item":
            row["impressions_raw"] = row["impressions"]
            row["impressions"] = row["impressions"].split("|")
            row["impressions_hash"] = "|".join(sorted(row["impressions"]))
            row["index_clicked"] = (
                row["impressions"].index(row["reference"]) if row["reference"] in row["impressions"] else -1000
            )
            row["prices"] = list(map(int, row["prices"].split("|")))
            row["price_clicked"] = row["prices"][row["index_clicked"]] if row["index_clicked"] >= 0 else 0
        return row

    def process_rows(self, acc, rows):
        output_rows = []
        for clickout_id, row in tqdm(enumerate(rows)):
            if row["action_type"] == "clickout item":
                for rank, (item_id, price) in enumerate(zip(row["impressions"], row["prices"])):
                    if acc:
                        obs = self.calculate_features_per_item(acc, clickout_id, item_id, price, rank, row)
                    else:
                        obs = row.copy()
                        obs["item_id"] = item_id
                        obs["item_id_clicked"] = row["reference"]
                        obs["was_clicked"] = int(row["reference"] == item_id)
                        obs["clickout_id"] = clickout_id
                        obs["rank"] = rank
                        obs["price"] = price
                        del obs["fake_impressions"]
                        del obs["fake_impressions_raw"]
                        del obs["fake_prices"]
                        del obs["impressions"]
                        del obs["impressions_hash"]
                        del obs["impressions_raw"]
                        del obs["prices"]
                        del obs["action_type"]
                    output_rows.append(obs)
        return output_rows


@click.command()
@click.option("--limit", type=int, help="Number of rows to process")
def main(limit):
    accumulators, accs_by_action_type = get_accumulators_basic()
    data_io = DataIOChunks()
    feature_generator = FeatureGenerator(
        data_io=data_io,
        limit=limit, accumulators=accumulators, accs_by_action_type=accs_by_action_type
    )
    feature_generator.generate_features()


if __name__ == "__main__":
    main()
