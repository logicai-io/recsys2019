from csv import DictReader, DictWriter

import click
from recsys.data_generator.accumulators import logger, get_accumulators
from recsys.data_generator.data_io import DataIO
from tqdm import tqdm


class FeatureGenerator:
    def __init__(self, data_io: DataIO, limit, accumulators, accs_by_action_type):
        self.data_io = data_io
        self.limit = limit
        self.accumulators = accumulators
        self.accs_by_action_type = accs_by_action_type
        print("Number of accumulators %d" % len(self.accumulators))

    def calculate_features_per_item(self, clickout_id, item_id, price, rank, row):
        obs = row.copy()
        obs["item_id"] = item_id
        obs["item_id_clicked"] = row["reference"]
        obs["was_clicked"] = int(row["reference"] == item_id)
        obs["clickout_id"] = clickout_id
        obs["rank"] = rank
        obs["price"] = price
        self.update_obs_with_acc(obs, row)
        del obs["fake_impressions"]
        del obs["fake_impressions_raw"]
        del obs["fake_prices"]
        del obs["impressions"]
        del obs["impressions_hash"]
        del obs["impressions_raw"]
        del obs["prices"]
        del obs["action_type"]
        return obs

    def update_obs_with_acc(self, obs, row):
        for acc in self.accumulators:
            value = acc.get_stats(row, obs)
            if isinstance(value, dict):
                for k, v in value.items():
                    obs[k] = v
            else:
                obs[acc.name] = acc.get_stats(row, obs)

    def generate_features(self):
        for clickout_id, row in tqdm(enumerate(self.data_io.rows())):
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
                for rank, (item_id, price) in enumerate(zip(row["impressions"], row["prices"])):
                    obs = self.calculate_features_per_item(clickout_id, item_id, price, rank, row)
                    self.data_io.writer_queue.put(obs)

            for acc in self.accs_by_action_type[row["action_type"]]:
                acc.update_acc(row)


@click.command()
@click.option("--limit", type=int, help="Number of rows to process")
def main(limit):
    data_io = DataIO(limit=limit)
    accumulators, accs_by_action_type = get_accumulators()
    feature_generator = FeatureGenerator(
        data_io=data_io,
        limit=limit, accumulators=accumulators, accs_by_action_type=accs_by_action_type
    )
    feature_generator.generate_features()


if __name__ == "__main__":
    main()
