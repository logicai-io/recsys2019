package recsys

import recsys.Types.{ItemId, Price}

object Types {
  type ItemId     = Int
  type Price      = Int
  type UserId     = String
  type SessionId  = String
  type ActionType = String
  type Timestamp  = Int
  val DummyItem: ItemId = -1
}

case class Item(itemId: ItemId, price: Price)
