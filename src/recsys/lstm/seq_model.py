import json

from recsys.lstm.seq_utils import SeqVectorizer, calc_mrr, create_model
from sklearn.preprocessing import LabelBinarizer

with open("lstm_data.ndjson", encoding="utf-8") as inp:
    data = list(map(json.loads, inp.readlines()))

PAD = 25

cutoff = len(data) // 2
train_data = data[:cutoff]
val_data = data[cutoff:]
vect = SeqVectorizer(pad=PAD)
vect.fit(train_data)

train_data_tr = vect.transform(train_data)
val_data_tr = vect.transform(val_data)

model = create_model(vect)

lb_encoder = LabelBinarizer()

y_train_raw = [int(r["index_clicked"]) if r["index_clicked"] != "UNK" else 30 for r in train_data]
y_val_raw = [int(r["index_clicked"]) if r["index_clicked"] != "UNK" else 30 for r in val_data]

y_train_enc = lb_encoder.fit_transform(y_train_raw)
y_val_enc = lb_encoder.transform(y_val_raw)

model.fit(train_data_tr, y_train_enc, validation_data=(val_data_tr, y_val_enc), epochs=20, batch_size=512, shuffle=True)

train_preds = model.predict(train_data_tr, batch_size=512, verbose=True)
val_preds = model.predict(val_data_tr, batch_size=512, verbose=True)

mrr_val = calc_mrr(val_preds, y_val_enc)
mrr_train = calc_mrr(train_preds, y_train_enc)
