package recsys.accumulators

import java.io.File

import com.univocity.parsers.csv.{CsvParser, CsvParserSettings}
import recsys.Types.ItemId
import recsys.{CSVReader, Item, Row}

import scala.collection.JavaConversions._
import scala.collection.mutable

class ItemOriginality extends AccumulatorClickout {
  private val itemProperties = readProperties("../data/item_metadata.csv")

  override def update(row: Row): Unit = {}

  override def getStats(row: Row, items: Array[Item]): Array[mutable.LinkedHashMap[String, Any]] = {
    val fs = Array[mutable.LinkedHashMap[String, Any]]()
    val allProperties = items.map(item => itemProperties(item.itemId))
    allProperties.map(
      thisItem => {
        val f = mutable.LinkedHashMap[String, Any]()
        val avgSim = allProperties.map(otherItem => thisItem.intersect(otherItem).size).sum.toFloat / allProperties.length
        f("avg_properties_similarity") = avgSim
        f("avg_properties_similarity_norm") = avgSim / thisItem.size
        f
      }
    )
  }

  private def readProperties(path: String) = {
    val itemPropertiesReader = CSVReader.getCSVReader(path)
    val itemProperties =
      mutable.OpenHashMap[ItemId, mutable.BitSet]().withDefaultValue(mutable.BitSet())
    val propMap = mutable.Map[String, Int]()
    for (row <- itemPropertiesReader) {
      val itemId     = row.getInt("item_id")
      val properties = row.getString("properties").split('|')
      for (prop <- properties) {
        if (!propMap.contains(prop)) {
          propMap(prop) = propMap.size
        }
      }
      itemProperties(itemId) = mutable.BitSet(properties.map(propMap): _*)
    }
    itemProperties
  }
}
