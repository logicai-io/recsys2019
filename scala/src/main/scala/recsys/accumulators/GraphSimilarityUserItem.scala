package recsys.accumulators

import recsys.Types.{ItemId, Timestamp, UserId}
import recsys.{Item, Row}

import scala.collection.mutable

class GraphSimilarityUserItem(featureName: String, actionTypes: List[String]) extends Accumulator {

  private var ItemsUserMap                  = mutable.Map[ItemId, mutable.Set[UserId]]()
  private var UserItemsMap                  = mutable.Map[UserId, mutable.Set[ItemId]]()
  private var cacheKey: (UserId, Timestamp) = ("", 0)
  private var cacheVal                      = mutable.Map[ItemId, Int]()

  override def update(row: Row): Unit = {
    if (actionTypes.contains(row.actionType) && (row.isTest == 0) && (row.referenceItem != 0)) {
      if (!ItemsUserMap.contains(row.referenceItem)) {
        ItemsUserMap(row.referenceItem) = mutable.Set(row.userId)
      } else {
        ItemsUserMap(row.referenceItem).add(row.userId)
      }

      if (!UserItemsMap.contains(row.userId)) {
        UserItemsMap(row.userId) = mutable.Set(row.referenceItem)
      } else {
        UserItemsMap(row.userId).add(row.referenceItem)
      }
    }
  }

  override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
    val f     = mutable.LinkedHashMap[String, Any]()

    val key = (row.userId, row.timestamp)
    if (key == cacheKey) {
      f(featureName) = cacheVal.getOrElse(item.itemId, 0)
      return f
    }

    val items = mutable.Map[ItemId, Int]().withDefaultValue(0)
    for (itemId <- UserItemsMap.getOrElse(row.userId, List())) {
      for (userId <- ItemsUserMap.getOrElse(itemId, List())) {
        if (userId != row.userId) {
          for (itemId2 <- UserItemsMap.getOrElse(userId, List())) {
            items(itemId2) += 1
          }
        }
      }
    }
    f(featureName) = items.getOrElse(item.itemId, 0)

    cacheKey = key
    cacheVal = items
    f
  }
}
