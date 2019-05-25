package recsys.accumulators

import recsys.Types.{ItemId, UserId}
import recsys.{Item, Row}

import scala.collection.mutable

/*
class SimilarUsersItemInteraction:
    """
    This is an accumulator that given interaction with items
    Finds users who interacted with the same items and then gathers statistics of interaction
    from them
    """

    def __init__(self):
        self.action_types = ACTIONS_WITH_ITEM_REFERENCE
        self.items_users = defaultdict(set)
        self.users_items = defaultdict(set)
        self.cache_key = None
        self.item_stats_cached = None

    def update_acc(self, row):
        self.items_users[row["reference"]].add(row["user_id"])
        self.users_items[row["user_id"]].add(row["reference"])

    def get_stats(self, row, item):
        items_stats = self.read_stats_from_cache(row)
        obs = {}
        obs["similar_users_item_interaction"] = items_stats[item["item_id"]]
        return obs

    def read_stats_from_cache(self, row):
        key = (row["user_id"], row["timestamp"])
        if self.cache_key == key:
            items_stats = self.item_stats_cached
        else:
            items_stats = self.get_items_stats(row)
            self.item_stats_cached = items_stats
            self.cache_key = key
        return items_stats

    def get_items_stats(self, row):
        items = defaultdict(int)
        for item_id in self.users_items[row["user_id"]]:
            for user_id in self.items_users[item_id]:
                # discard the self similarity
                if user_id == row["user_id"]:
                    continue
                for item_id_2 in self.users_items[user_id]:
                    items[item_id_2] += 1
        return items
 */

class Accumulator {
    def update(row: Row, item: Item): Unit = {

    }

    def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
        val f = mutable.LinkedHashMap[String, Any]()
        f
    }
}

class GraphSimilarity extends Accumulator {

    private var ItemsUserMap = mutable.Map[ItemId, mutable.Set[UserId]]()
    private var UserItemsMap = mutable.Map[UserId, mutable.Set[ItemId]]()

    override def update(row: Row, item: Item): Unit = {

    }

    override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
        val f = mutable.LinkedHashMap[String, Any]()

        f
    }
}
