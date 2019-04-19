def increment_key_by_one(acc, key):
    acc[key] += 1
    return acc


def increment_keys_by_one(acc, keys):
    for key in keys:
        acc[key] += 1
    return acc


def add_to_set(acc, key, value):
    acc[key].add(value)
    return True


def set_key(acc, key, value):
    acc[key] = value
    return True


def set_nested_key(acc, key1, key2, value):
    acc[key1][key2] = value
    return acc


def add_one_nested_key(acc, key1, key2):
    acc[key1][key2] += 1
    return acc


def append_to_list(acc, key, value):
    acc[key].append(value)
    return True


def append_to_list_not_null(acc, key, value):
    if value:
        acc[key].append(value)
    return True


def diff_ts(acc, current_ts):
    new_acc = acc.copy()
    for key in new_acc.keys():
        new_acc[key] = current_ts - acc[key]
    return new_acc


def tryint(s):
    try:
        return int(s)
    except:
        return 0


def diff(seq):
    new_seq = []
    for n in range(len(seq) - 1):
        new_seq.append(seq[n + 1] - seq[n])
    return new_seq
