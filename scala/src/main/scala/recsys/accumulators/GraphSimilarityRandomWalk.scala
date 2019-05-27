package recsys.accumulators

import recsys.Types.{ItemId, Timestamp, UserId}
import recsys.{Item, Row}

import scala.collection.mutable
import scala.util.Random

class GraphSimilarityRandomWalk(featureName: String, actionTypes: List[String]) extends AccumulatorClickout {

  private var ItemsUserMap = mutable.Map[ItemId, mutable.Set[UserId]]()
  private var UserItemsMap = mutable.Map[UserId, mutable.Set[ItemId]]()

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

  override def getStats(row: Row, items: Array[Item]): Array[mutable.LinkedHashMap[String, Any]] = {
    var currentUser = row.userId
    var currentItem = 0
    val itemsWeight = mutable.Map[ItemId, Int]().withDefaultValue(0)
    if (UserItemsMap.contains(currentUser)) {
      for (n <- 1 to 1000) {
        val userItems = UserItemsMap(currentUser)
        currentItem = userItems.toList(Random.nextInt(userItems.size))
        val users = ItemsUserMap(currentItem)
        currentUser = users.toList(Random.nextInt(users.size))
        if (currentUser != row.userId) {
          itemsWeight(currentItem) += 1
        }
      }
    }
    val fs = items.map(item => mutable.LinkedHashMap[String, Any](featureName -> itemsWeight(item.itemId)))
    fs
  }
}
