package recsys

import java.io.{File, StringWriter}

import com.univocity.parsers.common.record.Record
import com.univocity.parsers.csv.{CsvParser, CsvParserSettings, CsvWriter, CsvWriterSettings}
import me.tongfei.progressbar.ProgressBar
import recsys.Types._

import scala.collection.JavaConversions._
import scala.collection.mutable
import recsys.accumulators.{Accumulator, AccumulatorClickout, GraphSimilarityRandomWalk, GraphSimilarityRandomWalkResets, GraphSimilarityUserItem, GraphSimilarityUserItemAction, MetaFeaturesAll, MetaFeaturesId, MetaFeaturesSmall}

class GenerateFeaturesAcc(inputPath: String,
                          outputPath: String,
                          accumulators: List[Accumulator] = List(),
                          accumulatorsClickout: List[AccumulatorClickout] = List(),
                          addClickout: Boolean = false) {

  type FeaturesMap = mutable.LinkedHashMap[String, Any]

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

  private val itemProperties = readProperties("../data/item_metadata.csv")

  def run() {
    val writer       = getWriter(outputPath)
    val eventsReader = getCSVReader(inputPath)

    val pb            = new ProgressBar("Calculating features", 19715327)
    var headerWritten = false
    for ((rawRow, clickoutId) <- eventsReader.zipWithIndex) {
      pb.step()
      val actionType = rawRow.getString("action_type")
      val row        = extractRowObj(rawRow, actionType)

      if (actionType == "clickout item") {
        val itemsFeatures = if (accumulators.nonEmpty) {
          row.items.map(item => extractFeatures(clickoutId, row, item))
        } else {
          extractFeaturesClickout(clickoutId, row, row.items)
        }
        if (!headerWritten) {
          writer.writeHeaders(
            itemsFeatures.head.keys.toList.map(normalizeColumnName)
          )
          headerWritten = true
        }
        writer.writeRows(
          itemsFeatures.map(f => f.values.toArray.map(_.asInstanceOf[AnyRef]))
        )
      }

      updateAccumulators(actionType, row)
    }
    writer.close()
    pb.close()
  }

  private def normalizeColumnName(colName: String): String =
    colName.replace(' ', '_')

  private def updateAccumulators(actionType: String, row: Row): Unit = {
    for (acc <- accumulatorsClickout) {
      acc.update(row)
    }
    for (acc <- accumulators) {
      acc.update(row)
    }
  }

  private def extractFeatures(clickoutId: ItemId, row: Row, item: Item) = {
    val j           = item.itemId
    val featuresRow = mutable.LinkedHashMap[String, Any]()
    if (addClickout) {
      featuresRow("clickout_id") = clickoutId
    }
    for (acc <- accumulators) {
      for ((k, v) <- acc.getStats(row, item)) {
        featuresRow(k) = v
      }
    }
    featuresRow
  }

  def mergeWith[K, V](xs: collection.mutable.LinkedHashMap[K, V], ys: collection.mutable.LinkedHashMap[K, V])(
      f: (V, V) => V): collection.mutable.LinkedHashMap[K, V] = {
    val ns = collection.mutable.LinkedHashMap[K, V]()
    (xs.keySet ++ ys.keySet).foreach { k =>
      if (!xs.isDefinedAt(k)) ns.update(k, ys(k))
      else if (!ys.isDefinedAt(k)) ns.update(k, xs(k))
      else ns.update(k, f(xs(k), ys(k)))
    }
    ns
  }

  def mergeArrays(a: Array[FeaturesMap], b: Array[FeaturesMap]): Array[FeaturesMap] = {
    a.zip(b).map { x =>
      mergeWith(x._1, x._2)((a, b) => a)
    }
  }

  private def extractFeaturesClickout(clickoutId: ItemId, row: Row, items: Array[Item]): Array[FeaturesMap] = {
    val allFeatures    = accumulatorsClickout.map(acc => acc.getStats(row, items))
    val featuresMerged = allFeatures.reduce(mergeArrays)
    featuresMerged
  }

  private def extractRowObj(rawRow: Record, actionType: String) = {
    val (impressions, prices, items) = extractImpressionsPricesItems(rawRow)
    var row = Row(
      actionType = actionType,
      userId = rawRow.getString("user_id"),
      sessionId = rawRow.getString("session_id"),
      step = rawRow.getInt("step"),
      timestamp = rawRow.getInt("timestamp"),
      platform = rawRow.getString("platform"),
      city = rawRow.getString("city"),
      device = rawRow.getString("device"),
      currentFilters = rawRow.getString("current_filters"),
      referenceItem = extractItemReference(rawRow),
      referenceOther = extractOtherReference(rawRow),
      src = rawRow.getString("src"),
      isTest = rawRow.getInt("is_test"),
      impressions = impressions,
      prices = prices,
      items = items
    )
    row
  }

  private def extractImpressionsPricesItems(rawRow: Record) = {
    if (rawRow.getString("action_type") == "clickout item") {
      val impressions = rawRow.getString("impressions").split('|').map(_.toInt)
      val prices      = rawRow.getString("prices").split('|').map(_.toInt)
      val items: Array[Item] =
        impressions.zip(prices).zipWithIndex.map {
          case ((item: ItemId, price: Price), rank: Int) =>
            Item(item, price, rank)
        }
      (impressions, prices, items)
    } else {
      val impressions = null
      val prices      = null
      val items       = null
      (impressions, prices, items)
    }
  }

  private def extractItemReference(rawRow: Record): Int = {
    if (ACTIONS_WITH_ITEM_REF.contains(rawRow.getString("action_type")) & Utils
          .isAllDigits(
            rawRow.getString("reference")
          )) {
      rawRow.getInt("reference")
    } else {
      0
    }
  }

  private def getWriter(filePath: String) = {
    val output   = new StringWriter
    val settings = new CsvWriterSettings
    settings.setHeaderWritingEnabled(true)
    val writer = new CsvWriter(new File(filePath), settings)
    writer
  }

  private def extractOtherReference(rawRow: Record): String = {
    if (!ACTIONS_WITH_ITEM_REF.contains(rawRow.getString("action_type"))) {
      rawRow.getString("reference")
    } else {
      "unk"
    }
  }

  private def getCSVReader(filePath: String) = {
    val settings = new CsvParserSettings()
    settings.setHeaderExtractionEnabled(true)
    val reader = new CsvParser(settings)
    val it     = reader.iterateRecords(new File(filePath), "UTF-8")
    it.iterator()
  }

  private def readProperties(path: String) = {
    val itemPropertiesReader = getCSVReader(path)
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

  private def propertiesFreq(
      properties: mutable.Map[Types.ItemId, Set[String]]
  ) = {
    properties.values.flatten.groupBy(identity).mapValues(_.size)
  }
}

object GenerateFeaturesAcc {
  val inputPath = "../data/events_sorted.csv"
  def main(args: Array[String]) {
    generateFeatures(
      "../data/features/graph_similarity.csv",
      List(
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_clickout", actionTypes = List(CLICKOUT)),
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_search",
                                    actionTypes = List(SEARCH_FOR_ITEM)),
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_interaction_info",
                                    actionTypes = List(INTERACTION_INFO)),
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_interaction_img",
                                    actionTypes = List(INTERACTION_IMG)),
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_intearction_deal",
                                    actionTypes = List(INTERACTION_DEAL)),
        new GraphSimilarityUserItem(featureName = "graph_similarity_user_item_all_interactions",
                                    actionTypes = ACTIONS_WITH_ITEM_REF)
      ),
      overwrite=true
    )
    generateFeatures("../data/features/_meta_feautres_all.csv", List(new MetaFeaturesAll()), addClickout = true)
    generateFeatures("../data/features/_meta_features_small.csv", List(new MetaFeaturesSmall()), addClickout = true)
    generateFeatures("../data/features/_meta_features_id.csv", List(new MetaFeaturesId()), addClickout = true)
    generateFeatures("../data/features/_meta_features_id.csv", List(new MetaFeaturesId()), addClickout = true)
    generateFeatures(
      "../data/features/graph_similarity_user_item_random_walk.csv",
      accumulatorsClickouts =
        List(new GraphSimilarityRandomWalk("graph_similarity_user_item_random_walk", actionTypes = ACTIONS_WITH_ITEM_REF)),
      overwrite = true
    )
    generateFeatures(
      "../data/features/graph_similarity_user_item_random_walk_resets.csv",
      accumulatorsClickouts =
        List(new GraphSimilarityRandomWalkResets("graph_similarity_user_item_random_walk_resets", actionTypes = ACTIONS_WITH_ITEM_REF))
    )
  }

  private def generateFeatures(outputPath: String,
                               accumulators: List[Accumulator] = List(),
                               accumulatorsClickouts: List[AccumulatorClickout] = List(),
                               addClickout: Boolean = false,
                               overwrite: Boolean = false): Unit = {
    if (new java.io.File(outputPath).exists && !overwrite) {
      println(f"Skipping $outputPath")
    } else {
      println(f"Generating $outputPath")
      val generateFeatures =
        new GenerateFeaturesAcc(inputPath, outputPath, accumulators, accumulatorsClickouts, addClickout)
      generateFeatures.run()

    }
  }
}
