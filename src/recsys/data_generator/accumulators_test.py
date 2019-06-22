from recsys.data_generator.accumulators import RockPaperScissors


def test_rps():
    acc = RockPaperScissors(k=2)

    acc.update_acc({"reference": "1", "user_id": "1", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "1", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "1", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "1", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "1", "index_clicked": 0})

    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 1})
    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 2})
    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 3})
    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 4})
    acc.update_acc({"reference": "1", "user_id": "2", "index_clicked": 5})

    acc.update_acc({"reference": "1", "user_id": "3", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "3", "index_clicked": 1})
    acc.update_acc({"reference": "1", "user_id": "3", "index_clicked": 2})
    acc.update_acc({"reference": "1", "user_id": "3", "index_clicked": 3})
    acc.update_acc({"reference": "1", "user_id": "3", "index_clicked": 4})

    acc.update_acc({"reference": "1", "user_id": "4", "index_clicked": 0})
    acc.update_acc({"reference": "1", "user_id": "4", "index_clicked": 1})
    acc.update_acc({"reference": "1", "user_id": "4", "index_clicked": 2})

    acc.update_acc({"reference": "1", "user_id": "5", "index_clicked": 0})

    acc.update_acc({"reference": "1", "user_id": "7", "index_clicked": 10})

    print(acc.items_indices)
    print(acc.chain)

    print(acc.get_stats({"user_id": "4"}, {"item_id": 1, "rank": 0}))
    print(acc.get_stats({"user_id": "4"}, {"item_id": 1, "rank": 3}))
    print(acc.get_stats({"user_id": "6"}, {"item_id": 1, "rank": 0}))
    print(acc.get_stats({"user_id": "7"}, {"item_id": 1, "rank": 0}))


if __name__ == '__main__':
    test_rps()