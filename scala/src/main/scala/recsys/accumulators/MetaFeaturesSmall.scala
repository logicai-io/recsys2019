package recsys.accumulators

import recsys.{Item, Row}

import scala.collection.mutable

class MetaFeatures extends Accumulator {
  override def update(row: Row): Unit = {

  }

  override def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
    val f = mutable.LinkedHashMap[String, Any]()
    f("src") = row.src
    f("is_test") = row.isTest
    f("was_clicked") = if (row.referenceItem == item.itemId) 1 else 0
    f
  }
}
