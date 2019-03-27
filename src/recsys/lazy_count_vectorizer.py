from collections import defaultdict

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, _make_int_array
from sklearn.utils.fixes import sp_version


class LazyCountVectorizer(CountVectorizer):
    def __init__(
        self,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
        document_map=None,
        default_value=None,
    ):
        super().__init__(
            input,
            encoding,
            decode_error,
            strip_accents,
            lowercase,
            preprocessor,
            tokenizer,
            stop_words,
            token_pattern,
            ngram_range,
            analyzer,
            max_df,
            min_df,
            max_features,
            vocabulary,
            binary,
            dtype,
        )
        self.document_map = document_map
        self.default_value = default_value
        self.dtype = dtype

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for id in raw_documents:
            doc = self.document_map.get(id, self.default_value or [])
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only" " contain stop words")

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            if sp_version >= (0, 14):
                indices_dtype = np.int64
            else:
                raise ValueError(
                    (
                        "sparse CSR array has {} non-zero "
                        "elements and requires 64 bit indexing, "
                        " which is unsupported with scipy {}. "
                        "Please upgrade to scipy >=0.14"
                    ).format(indptr[-1], ".".join(sp_version))
                )

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X


if __name__ == "__main__":
    vect = LazyCountVectorizer(document_map={1: "ala ma kota", 2: "ala ma psa", 3: "paweł ma laptopa"})
    X = vect.fit_transform([1, 2, 3])
    print(X.shape)
