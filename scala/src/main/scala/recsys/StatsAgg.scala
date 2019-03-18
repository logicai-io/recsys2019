package recsys

import com.univocity.parsers.common.record.Record

import scala.collection.mutable

class StatsAgg[K, V](
    val name: String,
    val filter: Row => Boolean,
    val keyFunc: Row => K,
    val updater: mutable.Map[K, V] => Unit
) {
  val acc: mutable.Map[K, V] = mutable.Map[K, V]()
}

object StatsAgg {}
/*
    StatsAcc(
        name="clickout_item_impressions",
        filter=lambda row: row["action_type"] == "clickout item",
        acc=defaultdict(int),
        updater=lambda acc, row: increment_keys_by_one(acc, row["impressions"]),
        get_stats_func=lambda acc, row, item: acc[item["item_id"]],
    ),
 */
