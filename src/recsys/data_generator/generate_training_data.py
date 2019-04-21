from csv import DictReader, DictWriter

import click
from recsys.data_generator.accumulators import get_accumulators, logger


class FeatureGenerator:
    def __init__(self, limit, accumulators, accs_by_action_type, save_only_features=False, save_as=None):
        self.limit = limit
        self.accumulators = accumulators
        self.accs_by_action_type = accs_by_action_type
        self.save_only_features = save_only_features
        self.save_as = save_as
        print("Number of accumulators %d" % len(self.accumulators))

    def calculate_features_per_item(self, clickout_id, item_id, price, rank, row):
        obs = row.copy()
        obs["item_id"] = item_id
        obs["item_id_clicked"] = row["reference"]
        obs["was_clicked"] = int(row["reference"] == item_id)
        obs["clickout_id"] = clickout_id
        obs["rank"] = rank
        obs["price"] = price
        features = self.update_obs_with_acc(obs, row)
        del obs["fake_impressions"]
        del obs["fake_impressions_raw"]
        del obs["fake_prices"]
        del obs["impressions"]
        del obs["impressions_hash"]
        del obs["impressions_raw"]
        del obs["prices"]
        del obs["action_type"]
        return obs, features

    def update_obs_with_acc(self, obs, row):
        features = []
        for acc in self.accumulators:
            value = acc.get_stats(row, obs)
            if isinstance(value, dict):
                for k, v in value.items():
                    obs[k] = v
                    features.append(k)
            else:
                obs[acc.name] = acc.get_stats(row, obs)
                features.append(acc.name)
        return features

    def generate_features(self):
        logger.info("Starting feature generation")
        rows_gen = self.read_rows()
        logger.info("Starting processing")
        output_obs_gen = self.process_rows(rows_gen)
        self.save_rows(output_obs_gen)

    def save_rows(self, output_obs):
        out = open(self.save_as, "wt")
        first_row = True
        for obs, features in output_obs:
            if first_row:
                if self.save_only_features:
                    dw = DictWriter(out, fieldnames=features, lineterminator="\n")
                else:
                    dw = DictWriter(out, fieldnames=obs.keys(), lineterminator="\n")
                dw.writeheader()
                first_row = False
            if self.save_only_features:
                obs = {k: v for k, v in obs.items() if k in features}
                dw.writerow(obs)
            else:
                dw.writerow(obs)
        out.close()

    def read_rows(self):
        inp = open("../../../data/events_sorted.csv")
        dr = DictReader(inp)
        print("Reading rows")
        for i, row in enumerate(dr):
            yield row
            if self.limit and i > self.limit:
                break
        inp.close()

    def process_rows(self, rows):
        for clickout_id, row in enumerate(rows):
            if clickout_id % 100000 == 0:
                print(self.save_as, clickout_id)
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
                    obs, features = self.calculate_features_per_item(clickout_id, item_id, price, rank, row)
                    yield obs, features

            for acc in self.accs_by_action_type[row["action_type"]]:
                acc.update_acc(row)


@click.command()
@click.option("--limit", type=int, help="Number of rows to process")
@click.option("--hashn", type=int, default=None, help="Chunk number")
def main(limit, hashn):
    print(hashn)
    save_as = "../../../data/events_sorted_trans_%02d.csv" % (hashn)
    accumulators, accs_by_action_type = get_accumulators(hashn)
    feature_generator = FeatureGenerator(
        limit=limit,
        accumulators=accumulators,
        accs_by_action_type=accs_by_action_type,
        save_only_features=hashn != 0,
        save_as=save_as,
    )
    feature_generator.generate_features()


if __name__ == "__main__":
    main()
