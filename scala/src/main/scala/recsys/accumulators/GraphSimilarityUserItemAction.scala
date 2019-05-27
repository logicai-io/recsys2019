package recsys.accumulators

import recsys.Types.{ActionType, ItemId, Timestamp, UserId}
import recsys.{Item, Row}

import scala.collection.mutable

class GraphSimilarityUserItemAction extends Accumulator {

  private var ItemsUserMap                  = mutable.Map[(ItemId, ActionType), mutable.Set[UserId]]()
  private var UserItemsMap                  = mutable.Map[UserId, mutable.Set[(ItemId, ActionType)]]()
  private var cacheKey: (UserId, Timestamp) = ("", 0)
  private var cacheVal                      = mutable.Map[ItemId, Int]()

  override def update(row: Row): Unit = {
    if (row.actionType == "clickout item" && (row.isTest == 0) && (row.referenceItem != 0)) {
      val key = (row.referenceItem, row.actionType)
      if (!ItemsUserMap.contains(key)) {
        ItemsUserMap(key) = mutable.Set(row.userId)
      } else {
        ItemsUserMap(key).add(row.userId)
      }

      if (!UserItemsMap.contains(row.userId)) {
        UserItemsMap(row.userId) = mutable.Set(key)
      } else {
        UserItemsMap(row.userId).add(key)
      }
    }
  }

  override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
    val f     = mutable.LinkedHashMap[String, Any]()

    val key = (row.userId, row.timestamp)
    if (key == cacheKey) {
      f("similar_users_item_action_type_interaction") = cacheVal.getOrElse(item.itemId, 0)
      return f
    }

    val items = mutable.Map[ItemId, Int]().withDefaultValue(0)
    for (itemId <- UserItemsMap.getOrElse(row.userId, List())) {
      for (userId <- ItemsUserMap.getOrElse(itemId, List())) {
        if (userId != row.userId) {
          for (itemId2 <- UserItemsMap.getOrElse(userId, List())) {
            items(itemId2._1) += 1
          }
        }
      }
    }
    f("similar_users_item_action_type_interaction") = items.getOrElse(item.itemId, 0)

    cacheKey = key
    cacheVal = items
    f
  }
}
