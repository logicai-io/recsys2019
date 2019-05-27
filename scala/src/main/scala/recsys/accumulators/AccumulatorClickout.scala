package recsys.accumulators

import recsys.{Item, Row}

import scala.collection.mutable

class AccumulatorClickout {
    def update(row: Row): Unit = {

    }

    def getStats(row: Row, items: Array[Item]): Array[mutable.LinkedHashMap[String, Any]] = {
        val fs = Array[mutable.LinkedHashMap[String, Any]]()
        items.map(
          item => {
            val f = mutable.LinkedHashMap[String, Any]()
            f("dummy") = 1
            f
          }
        )
    }
}
