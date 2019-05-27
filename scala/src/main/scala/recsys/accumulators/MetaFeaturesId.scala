package recsys.accumulators

import recsys.{Item, Row}

import scala.collection.mutable

class MetaFeaturesId extends Accumulator {
  override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
    val f = mutable.LinkedHashMap[String, Any]()
    f("user_id") = row.userId
    f("session_id") = row.sessionId
    f("step") = row.step
    f
  }
}
