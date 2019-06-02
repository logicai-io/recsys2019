from collections import Counter, deque
from itertools import chain, repeat
from typing import List, Iterable

import numpy as np

UNKNOWN_TOKEN = 'UNK'
UNKNOWN_TOKEN_POS = 0


def build_reversed_index(xs: List):
    return dict(zip(xs, range(len(xs))))


def get_most_common(seq, encode_unknown, size):
    counter = Counter(seq)
    if encode_unknown:
        most_common = [UNKNOWN_TOKEN]
    else:
        most_common = []
    most_common += [el for el, freq in counter.most_common()]
    if size:
        most_common = most_common[:size]
    return most_common


class BaseListVectorizer():
    def __init__(self, encode_unknown: bool = True, size=10000, onehot: bool = True):
        self.encode_unknown = encode_unknown
        self.size = size
        self.onehot = onehot

    def get_most_common(self, tokens_list):
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)
        if self.encode_unknown:
            most_common = [UNKNOWN_TOKEN]
        else:
            most_common = []
        most_common += [word for word, freq in counter.most_common()][:self.size - 1]
        return most_common

    @property
    def output_size(self):
        return len(self.idx)

    def transform_one_hot(self, tokens_list):
        pass

    def transform_indices(self, tokens_list):
        pass

    def transform(self, tokens_list):
        if self.onehot:
            return self.transform_one_hot(tokens_list)
        else:
            return self.transform_indices(tokens_list)


class ListVectorizer(BaseListVectorizer):
    """
    Class that transforms flat lists to indices.
    It can be used to vectorize words like this

    Example:

    >>> vectorizer = ListVectorizer(onehot=False)
    >>> tokens = ["this", "is", "a", "word"]
    >>> vectorizer.fit(tokens)
    [1, 2, 3, 4]

    """

    def __init__(self, encode_unknown: bool = True, size=10000, onehot: bool = True):
        super().__init__(encode_unknown, size)
        self.encode_unknown = encode_unknown
        self.size = size
        self.onehot = onehot

    def fit(self, tokens_list: List[str]):
        most_common = self.get_most_common([tokens_list])
        self.idx = build_reversed_index(most_common)
        return self

    def transform_indices(self, tokens) -> List[int]:
        output = []
        for token in tokens:
            output.append(self.idx.get(token, UNKNOWN_TOKEN_POS))
        return output

    def transform_one_hot(self, tokens_list) -> np.ndarray:
        data = np.zeros((len(tokens_list), len(self.idx)), dtype=np.float)
        for i, token in enumerate(tokens_list):
            data[i, self.idx.get(token, UNKNOWN_TOKEN_POS)] = 1
        return data

    def transform(self, tokens_list):
        if self.onehot:
            return self.transform_one_hot(tokens_list)
        else:
            return self.transform_indices(tokens_list)


class NestedListVectorizer(BaseListVectorizer):
    """
    A class to transform a list of iterables - it can be a list of tokens

    Example:

    >>> nested_vocabulary = NestedListVectorizer(onehot=False)
    >>> tokens = ["this", "is", "a", "word"]
    >>> nested_vocabulary.fit(tokens)
    >>> print(nested_vocabulary.transform(tokens))

    [[3, 4, 1, 2], [1, 2], [5], [6, 7, 8, 9]]
    """

    def __init__(self, encode_unknown: bool = True, size=10000, onehot: bool = True):
        super().__init__(encode_unknown, size)
        self.encode_unknown = encode_unknown
        self.size = size
        self.onehot = onehot

    def fit(self, tokens_nested_list: List[Iterable[str]]) -> 'NestedListVectorizer':
        most_common = self.get_most_common(list(chain(*tokens_nested_list)))
        self.idx = build_reversed_index(most_common)
        return self

    def transform_indices(self, tokens_nested_list) -> List[List[int]]:
        output = []
        for tokens in tokens_nested_list:
            output.append([self.idx.get(token, UNKNOWN_TOKEN_POS) for token in tokens])
        return output

    def transform_one_hot(self, tokens_nested_list):
        data = np.zeros((len(tokens_nested_list), len(self.idx)), dtype=np.float)
        for i, tokens_list in enumerate(tokens_nested_list):
            for token in tokens_list:
                data[i, self.idx.get(token, UNKNOWN_TOKEN_POS)] = 1
        return data


class DeepListVectorizer():
    def __init__(self, depth: int = 1, size: int = None, onehot: bool = True, encode_unknown: bool = True):
        self.depth = depth
        self.size = size
        self.onehot = onehot
        self.encode_unknown = encode_unknown

    def fit(self, seq: List) -> 'DeepListVectorizer':
        queue = deque(list(zip(repeat(0), seq)))
        elements = []
        while queue:
            depth, element = queue.popleft()
            if depth == self.depth:
                elements.append(element)
            else:
                for el in element:
                    queue.append((depth + 1, el))
        most_common = get_most_common(elements, self.encode_unknown, self.size)
        self.idx = build_reversed_index(most_common)
        return self

    def transform(self, seq: List) -> List:
        return self.rebuild(seq, depth=0)

    def build_vector(self, id):
        onehot = [0]*len(self.idx)
        onehot[id] = 1
        return onehot

    def rebuild(self, seq: List, depth: int) -> List:
        if depth == self.depth:
            output = []
            for el in seq:
                pos = self.idx.get(el, UNKNOWN_TOKEN_POS)
                if self.onehot:
                    output.append(self.build_vector(pos))
                else:
                    output.append(pos)
            return output
        elif depth < self.depth:
            output = []
            for el in seq:
                output.append(self.rebuild(el, depth + 1))
            return output
        else:
            raise ValueError('Wrong depth for list indexer')


if __name__ == '__main__':
    vectorizer = DeepListVectorizer(depth=1, onehot=False) #DeepListVectorizer(depth=1, onehot=False)
    tokens = [['growbots', '-', 'title', 'paragraph', '1', 'paragraph', '2'],
              ['growbots', '-', 'title', 'paragraph', '1', 'paragraph', '2'],
              ['growbots', '-', 'title', 'paragraph', '1', 'paragraph', '2']]
    vectorizer.fit(tokens)
    print(vectorizer.transform(tokens))