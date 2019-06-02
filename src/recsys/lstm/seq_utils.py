from collections import defaultdict

import keras as ks
import numpy as np
from recsys.lstm.seq_model import PAD, val_data
from recsys.lstm.seq_vectorizers import DeepListVectorizer
from recsys.metric import mrr_fast_v2
from tqdm import tqdm


class SeqVectorizer():

    def __init__(self, pad=25):
        self.pad = pad

    def fit(self, data):
        # fit static vectorizers
        self.vectorizers = {}
        for key in ["platform", "city"]:
            v = DeepListVectorizer(onehot=False, depth=0)
            self.vectorizers[key] = v
            self.vectorizers[key].fit([row[key] for row in data])

        # fit sequence vectorizers
        lists = defaultdict(list)
        values = defaultdict(set)
        for obs in tqdm(data):
            for seq in obs['sequences']:
                if seq.endswith('cat'):
                    lists[seq].append(obs['sequences'][seq])
                    for v in obs['sequences'][seq]:
                        values[seq].add(v)

        self.seq_vectorizers = {}
        for key in values:
            if len(values[key]) < 30:
                self.seq_vectorizers[key] = DeepListVectorizer(onehot=True)
            else:
                self.seq_vectorizers[key] = DeepListVectorizer(onehot=False, size=500)
            self.seq_vectorizers[key].fit(lists[key])

        return self

    def transform(self, data):
        X = defaultdict(list)

        for key in ["platform", "city"]:
            vectorized = self.vectorizers[key].transform([row[key] for row in data])
            X[key] = vectorized

        X["prices"] = np.log1p(np.array([(row["final_prices"] + [0]*25)[:25] for row in data]))

        for obs in tqdm(data):
            for key, vect in self.seq_vectorizers.items():
                if key.endswith("cat"):
                    if key not in ["action type cat",
                                   "clickout item cat",
                                   "interaction item image cat",
                                   "interaction item info cat",
                                   "interaction item rating cat",
                                   "interaction item deals cat",
                                   "change of sort order cat",
                                   "filter selection cat"]:
                        continue
                    vectorized = vect.transform([obs['sequences'][key]])
                    if vect.onehot:
                        X[key].append(([[-100]*len(vect.idx)]*self.pad + vectorized[0])[-self.pad:])
                    else:
                        X[key].append([-100]*self.pad + vectorized[0])

        for key in ["clickout item num"]:
            X[key] = [[([-100]*self.pad + row["sequences"][key])[-self.pad:]] for row in data]
            X[key] = np.array(X[key]).transpose(0,2,1)
            X[key] = X[key]/25

        for key in ["timestamp diff num"]:
            X[key] = [[([-100]*self.pad + row["sequences"][key])[-self.pad:]] for row in data]
            X[key] = np.array(X[key]).transpose(0,2,1)
            X[key] = np.log1p(X[key].clip(0, 180))

        for key in ["count action num"]:
            X[key] = [[([-100]*self.pad + row["sequences"][key])[-self.pad:]] for row in data]
            X[key] = np.array(X[key]).transpose(0,2,1)
            X[key] = np.log1p(X[key].clip(0, 50))

        X["device"] = np.array([row['device']=="desktop" for row in data], dtype=np.float).reshape(-1,1)

        X_ = {}
        for key in X:
            X_[key.replace(" ", "_")] = np.array(X[key])
        return X_


def calc_mrr(val_preds, y_val_enc):
    clickout_ids = []
    preds = []
    clicks = []
    for n in range(val_preds.shape[0]):
        n_items = len(val_data[n]["final_prices"])
        for item in range(n_items):
            clickout_ids.append(n)
            preds.append(val_preds[n, item])
            clicks.append(y_val_enc[n, item])
    return mrr_fast_v2(clicks, preds, clickout_ids)


def create_model(vect):
    platform_input = ks.layers.Input(shape=[1], name='platform')
    device_input = ks.layers.Input(shape=[1], name='device')
    city_input = ks.layers.Input(shape=[1], name='city')
    prices_input = ks.layers.Input(shape=[25], name='prices')

    seq_actiontype_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["action type cat"].idx)], name="action_type_cat")
    seq_clickout_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["clickout item cat"].idx)], name="clickout_item_cat")
    seq_int_img_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["interaction item image cat"].idx)], name="interaction_item_image_cat")
    seq_int_info_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["interaction item info cat"].idx)], name="interaction_item_info_cat")
    seq_int_rating_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["interaction item rating cat"].idx)], name="interaction_item_rating_cat")
    seq_int_deals_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["interaction item deals cat"].idx)], name="interaction_item_deals_cat")
    seq_change_sort_cat_input = ks.layers.Input(shape=[PAD, len(vect.seq_vectorizers["change of sort order cat"].idx)], name="change_of_sort_order_cat")

    seq_clickout_num_input = ks.layers.Input(shape=[PAD, 1], name="clickout_item_num")
    seq_timestamp_diff_num_input = ks.layers.Input(shape=[PAD, 1], name="timestamp_diff_num")
    seq_count_action_num_input = ks.layers.Input(shape=[PAD, 1], name="count_action_num")

    concat_seq = ks.layers.concatenate([
        seq_actiontype_cat_input,
        seq_clickout_cat_input,
        seq_int_img_cat_input,
        seq_int_info_cat_input,
        seq_int_rating_cat_input,
        seq_int_deals_cat_input,
        seq_change_sort_cat_input,
        seq_clickout_num_input,
        seq_timestamp_diff_num_input,
        seq_count_action_num_input
    ])

    masked = ks.layers.Masking(mask_value=-100.)(concat_seq)
    rnn_layer1 = ks.layers.LSTM(64)(masked)

    platform_emb = ks.layers.Embedding(len(vect.vectorizers["platform"].idx), 5)(platform_input)
    city_emb = ks.layers.Embedding(len(vect.vectorizers["city"].idx), 5)(city_input)

    concat = ks.layers.concatenate([
        ks.layers.Flatten()(platform_emb),
        prices_input,
        device_input,
        rnn_layer1
    ])

    dense_1 = ks.layers.Dense(128, activation='relu')(concat)
    drop_1 = ks.layers.Dropout(0.25)(dense_1)
    dense_2 = ks.layers.Dense(128, activation='relu')(drop_1)
    output = ks.layers.Dense(26, activation="softmax")(dense_2)

    model = ks.Model([platform_input,
                      device_input,
                      city_input,
                      prices_input,
                      seq_actiontype_cat_input,
                      seq_clickout_num_input,
                      seq_clickout_cat_input,
                      seq_int_img_cat_input,
                      seq_int_info_cat_input,
                      seq_int_rating_cat_input,
                      seq_int_deals_cat_input,
                      seq_change_sort_cat_input,
                      seq_timestamp_diff_num_input,
                      seq_count_action_num_input], output)

    opt = ks.optimizers.Adam(lr=0.003, decay=0.000001)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model