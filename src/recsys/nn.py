import keras as ks
import tensorflow as tf


def nn_fit_predict(xs, y_train):
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1, activation="sigmoid")(out)
        model = ks.Model(model_in, out)
        model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(4):
            model.fit(x=X_train, y=y_train, batch_size=2 ** (10 + min(i,3)), epochs=1, verbose=1)
        return model.predict(X_test, batch_size=1024)[:, 0]