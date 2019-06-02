import gzip
import json

import pandas as pd
from recsys.lstm.seq_utils import SeqVectorizer, calc_mrr, create_model, prepare_predictions
from sklearn.preprocessing import LabelBinarizer


def concat_oof_predictions():
    train_path_a = "../../../data/lstm/seq_user_session_train_hash_0.ndjson.gzip"
    val_path_a = "../../../data/lstm/seq_user_session_train_hash_1.ndjson.gzip"
    test_path_a = "../../../data/lstm/seq_user_session_test_hash_1.ndjson.gzip"
    test_preds_df_a, val_preds_df_a = oof_seq_predictions(test_path_a, train_path_a, val_path_a)

    train_path_b = "../../../data/lstm/seq_user_session_train_hash_1.ndjson.gzip"
    val_path_b = "../../../data/lstm/seq_user_session_train_hash_0.ndjson.gzip"
    test_path_b = "../../../data/lstm/seq_user_session_test_hash_0.ndjson.gzip"
    test_preds_df_b, val_preds_df_b = oof_seq_predictions(test_path_b, train_path_b, val_path_b)

    all_preds_df = pd.concat([val_preds_df_a, test_preds_df_a, val_preds_df_b, test_preds_df_b], axis=0)
    return all_preds_df


def concat_oof_predictions_user():
    train_path_a = "../../../data/lstm/seq_user_train_hash_0.ndjson.gzip"
    val_path_a = "../../../data/lstm/seq_user_train_hash_1.ndjson.gzip"
    test_path_a = "../../../data/lstm/seq_user_test_hash_1.ndjson.gzip"
    test_preds_df_a, val_preds_df_a = oof_seq_predictions(test_path_a, train_path_a, val_path_a)

    train_path_b = "../../../data/lstm/seq_user_train_hash_1.ndjson.gzip"
    val_path_b = "../../../data/lstm/seq_user_train_hash_0.ndjson.gzip"
    test_path_b = "../../../data/lstm/seq_user_test_hash_0.ndjson.gzip"
    test_preds_df_b, val_preds_df_b = oof_seq_predictions(test_path_b, train_path_b, val_path_b)

    all_preds_df = pd.concat([val_preds_df_a, test_preds_df_a, val_preds_df_b, test_preds_df_b], axis=0)
    return all_preds_df


def oof_seq_predictions(test_path, train_path, val_path):
    with gzip.open(train_path) as inp:
        train_data = list(map(json.loads, inp.readlines()))
    with gzip.open(val_path) as inp:
        val_data = list(map(json.loads, inp.readlines()))
    with gzip.open(test_path) as inp:
        test_data = list(map(json.loads, inp.readlines()))
    PAD = 25
    vect = SeqVectorizer(pad=PAD)
    vect.fit(train_data)
    train_data_tr = vect.transform(train_data)
    val_data_tr = vect.transform(val_data)
    test_data_tr = vect.transform(test_data)
    model = create_model(vect)
    lb_encoder = LabelBinarizer()
    y_train_raw = [int(r["index_clicked"]) if r["index_clicked"] != "UNK" else 30 for r in train_data]
    y_val_raw = [int(r["index_clicked"]) if r["index_clicked"] != "UNK" else 30 for r in val_data]
    y_train_enc = lb_encoder.fit_transform(y_train_raw)
    y_val_enc = lb_encoder.transform(y_val_raw)
    model.fit(train_data_tr, y_train_enc, validation_data=(val_data_tr, y_val_enc), epochs=5, batch_size=256,
              shuffle=True)
    train_preds = model.predict(train_data_tr, batch_size=512, verbose=True)
    val_preds = model.predict(val_data_tr, batch_size=512, verbose=True)
    test_preds = model.predict(test_data_tr, batch_size=512, verbose=True)
    mrr_val = calc_mrr(val_data, val_preds, y_val_enc)
    mrr_train = calc_mrr(train_data, train_preds, y_train_enc)
    print(f"MRR TRAIN {mrr_train:.4f} VAL {mrr_val:.4f}")
    val_preds_df = prepare_predictions(val_data, val_preds)
    test_preds_df = prepare_predictions(test_data, test_preds)
    return test_preds_df, val_preds_df


if __name__ == '__main__':
    oof_predictions = concat_oof_predictions()
    oof_predictions.to_csv("../../../data/lstm/oof_predictions_user_session.csv", index=False)

    oof_predictions = concat_oof_predictions_user()
    oof_predictions.to_csv("../../../data/lstm/oof_predictions_user.csv", index=False)
