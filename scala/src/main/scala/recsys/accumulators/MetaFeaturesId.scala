package recsys.accumulators

import recsys.{Item, Row}

import scala.collection.mutable

class MetaFeaturesAll extends Accumulator {
  override def update(row: Row): Unit = {

  }

  override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
    val f = mutable.LinkedHashMap[String, Any]()
    f("src") = row.src
    f("is_test") = row.isTest
    f("user_id") = row.userId
    f("session_id") = row.sessionId
    f("step") = row.step
    f("timestamp") = row.timestamp
    f("platform") = row.platform
    f("city") = row.city
    f("device") = row.device
    f("current_filters") = row.currentFilters
    f("reference") = row.referenceItem
    f("item_id") = item.itemId
    f("price") = item.price
    f("rank") = item.rank
    f("was_clicked") = if (row.referenceItem == item.itemId) 1 else 0
    f
  }
}
