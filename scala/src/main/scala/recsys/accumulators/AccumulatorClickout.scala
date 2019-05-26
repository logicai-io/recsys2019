package recsys.accumulators

import recsys.Types.{ItemId, UserId}
import recsys.{Item, Row}

import scala.collection.mutable

class Accumulator {
    def update(row: Row): Unit = {

    }

    def getStats(row: Row, item: Item): mutable.LinkedHashMap[String, Any] = {
        val f = mutable.LinkedHashMap[String, Any]()
        f
    }
}
