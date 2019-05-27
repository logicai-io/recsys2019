package recsys

import recsys.Types.{ItemId, Price}

import scala.collection.mutable

object Types {
  type ItemId     = Int
  type Price      = Int
  type UserId     = String
  type SessionId  = String
  type ActionType = String
  type Timestamp  = Int
  val DummyItem: ItemId = -1

  val ACTION_TYPES = List(
    "change of sort order",
    "clickout item",
    "filter selection",
    "interaction item deals",
    "interaction item image",
    "interaction item info",
    "search for destination",
    "search for item",
    "search for poi"
  )

  val ACTIONS_WITH_ITEM_REF = List(
    "search for item",
    "interaction item info",
    "interaction item image",
    "interaction item deals",
    "clickout item"
  )

  val SEARCH_FOR_ITEM = "search for item"
  val INTERACTION_INFO = "interaction item info"
  val INTERACTION_IMG = "interaction item image"
  val INTERACTION_DEAL = "interaction item deals"
  val CLICKOUT = "clickout item"


  type FeatureMap = mutable.LinkedHashMap[String, Any]
}

case class Item(itemId: ItemId, price: Price, rank: Int)

