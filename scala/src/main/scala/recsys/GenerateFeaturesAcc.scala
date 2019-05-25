package recsys

import java.io.{File, StringWriter}

import com.univocity.parsers.common.record.Record
import com.univocity.parsers.csv.{CsvParser, CsvParserSettings, CsvWriter, CsvWriterSettings}
import me.tongfei.progressbar.ProgressBar
import recsys.Types._

import scala.collection.JavaConversions._
import scala.collection.mutable

object GenerateFeatures {

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

  private var itemImpressions = mutable.Map[ItemId, Int]().withDefaultValue(0)
  private var itemClicks      = mutable.Map[ItemId, Int]().withDefaultValue(0)
  private var itemUserImpressions =
    mutable.Map[(ItemId, UserId), Int]().withDefaultValue(0)
  private var itemUserClicks =
    mutable.Map[(ItemId, UserId), Int]().withDefaultValue(0)
  private var lastSortOrder =
    mutable.Map[UserId, String]().withDefaultValue("unk")
  private var lastFilter = mutable.Map[UserId, String]().withDefaultValue("unk")
  private var lastPoi    = mutable.Map[UserId, String]().withDefaultValue("unk")
  private var itemsWithUserInteractionsSet =
    mutable.Map[UserId, mutable.ArrayBuffer[ItemId]]()
  private var itemsWithUserSessionInteractionsSet =
    mutable.Map[(UserId, SessionId), mutable.ArrayBuffer[ItemId]]()
  private var actionTypesTimestamps =
    mutable.Map[(UserId, ActionType), Timestamp]()
  private var actionTypesItemTimestamps =
    mutable.Map[(UserId, ItemId, ActionType), Timestamp]()
  private var actionTypesCounter =
    mutable.Map[(UserId, ActionType), Int]().withDefaultValue(0)
  private var actionTypesItemCounter =
    mutable.Map[(UserId, ItemId, ActionType), Int]().withDefaultValue(0)
  private var lastRefItemByActionType =
    mutable.Map[(UserId, ActionType), ItemId]().withDefaultValue(DummyItem)

  private val itemProperties = readProperties("../data/item_metadata.csv")
//  private val propFreq       = propertiesFreq(itemProperties)

  def main(args: Array[String]) {
//    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(8))

    val eventsReader = getCSVReader("../data/events_sorted_250k.csv")
    val writer       = getWriter("../data/events_sorted_trans_scala.csv")

    val pb            = new ProgressBar("Calculating features", 19715327)
    var headerWritten = false
    for ((rawRow, clickoutId) <- eventsReader.zipWithIndex) {
      pb.step()
      val actionType = rawRow.getString("action_type")
      val row        = extractRowObj(rawRow, actionType)

      if (actionType == "clickout item") {
//        val itemsPar = row.items.par
//        itemsPar.tasksupport = taskSupport
        val itemsFeatures =
          row.items.map(item => extractFeatures(clickoutId, row, item))
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
//    taskSupport.forkJoinPool.shutdown()
  }

  private def normalizeColumnName(colName: String): String =
    colName.replace(' ', '_')

  private def updateAccumulators(actionType: String, row: Row): Unit = {
    if (row.actionType == "clickout item") {
      itemClicks(row.referenceItem) += 1
      for (itemId <- row.impressions) {
        itemImpressions(itemId) += 1
      }
    }

    if (ACTIONS_WITH_ITEM_REF.contains(actionType)) {
      updateOrCreateSetMap(
        itemsWithUserInteractionsSet,
        row.userId,
        row.referenceItem
      )
      updateOrCreateSetMap(
        itemsWithUserSessionInteractionsSet,
        (row.userId, row.sessionId),
        row.referenceItem
      )
      actionTypesItemTimestamps(
        (row.userId, row.referenceItem, row.actionType)
      ) = row.timestamp
      actionTypesItemCounter((row.userId, row.referenceItem, row.actionType)) += 1
      lastRefItemByActionType((row.userId, row.actionType)) = row.referenceItem
    }

    if (row.actionType == "change of sort order") {
      lastSortOrder(row.userId) = row.referenceOther
    }

    if (row.actionType == "filter selection") {
      lastFilter(row.userId) = row.referenceOther
    }

    if (row.actionType == "search for poi") {
      lastPoi(row.userId) = row.referenceOther
    }

    actionTypesTimestamps((row.userId, row.actionType)) = row.timestamp
    actionTypesCounter((row.userId, row.actionType)) += 1
  }

  private def extractFeatures(clickoutId: ItemId, row: Row, item: Item) = {
    val j           = item.itemId
    val featuresRow = mutable.LinkedHashMap[String, Any]()
    featuresRow("clickout_id") = clickoutId
    featuresRow("src") = row.src
    featuresRow("is_test") = row.isTest
    featuresRow("user_id") = row.userId
    featuresRow("session_id") = row.sessionId
    featuresRow("step") = row.step
    featuresRow("timestamp") = row.timestamp
    featuresRow("platform") = row.platform
    featuresRow("city") = row.city
    featuresRow("device") = row.device
    featuresRow("current_filters") = row.currentFilters
    featuresRow("reference") = row.referenceItem
    featuresRow("item_id") = item.itemId
    featuresRow("price") = item.price
    featuresRow("rank") = item.rank
    featuresRow("was_clicked") = if (row.referenceItem == item.itemId) 1 else 0

    // get stats
    featuresRow("item_impressions") = itemImpressions(item.itemId)
    featuresRow("item_clicks") = itemClicks(item.itemId)
    featuresRow("item_ctr") = itemClicks(item.itemId).toFloat / (itemImpressions(
      item.itemId
    ) + 1)
    val itemUser = (item.itemId, row.userId)
    featuresRow("item_user_impressions") = itemUserImpressions(itemUser)
    featuresRow("item_user_clicks") = itemUserClicks(itemUser)
    featuresRow("item_user_ctr") = itemUserClicks(itemUser).toFloat / (itemUserImpressions(
      itemUser
    ) + 1)

    featuresRow("last_sort_order") = lastSortOrder(row.userId)
    featuresRow("last_filter_selection") = lastFilter(row.userId)
    featuresRow("last_poi") = lastPoi(row.userId)

    featuresRow("avg_jaccard_to_previous_interactions") = {
      calcAvgItemSim(itemsWithUserInteractionsSet, row.userId, item)
    }
    featuresRow("avg_jaccard_to_previous_interactions_session") = calcAvgItemSim(
      itemsWithUserSessionInteractionsSet,
      (row.userId, row.sessionId),
      item
    )

    for (actionType <- ACTION_TYPES) {
      featuresRow(s"${actionType}_last_timestamp") = math.min(
        row.timestamp -
          actionTypesTimestamps.getOrElse((row.userId, actionType), 0),
        1000000
      )
      featuresRow(s"${actionType}_count") = actionTypesCounter(
        (row.userId, actionType)
      )
    }

    for (actionType <- ACTIONS_WITH_ITEM_REF) {
      featuresRow(s"${actionType}_item_last_timestamp") = math.min(
        row.timestamp -
          actionTypesItemTimestamps
            .getOrElse((row.userId, item.itemId, actionType), 0),
        1000000
      )
      featuresRow(s"${actionType}_item_count") = actionTypesItemCounter(
        (row.userId, item.itemId, actionType)
      )
      featuresRow(s"${actionType}_last_ref_jaccard") = calcItemSim(
        lastRefItemByActionType((row.userId, actionType)),
        item.itemId
      )
      featuresRow(s"${actionType}_same_last_item") =
        if (lastRefItemByActionType((row.userId, actionType)) == item.itemId) 1
        else 0
    }

    featuresRow
  }

  private def calcItemSim(
      itemA: ItemId,
      itemB: ItemId,
      freq: Boolean = false
  ) = {
    if (itemA == DummyItem) {
      0.0
    } else {
      val itemAProp = itemProperties.getOrElse(itemA, Set[Int]())
      if (itemAProp.nonEmpty) {
        val itemBProp = itemProperties.getOrElse(itemB, Set[Int]())
        if (itemBProp.nonEmpty) {
          val intersection = itemAProp intersect itemBProp
          val union        = itemAProp union itemBProp
          if (freq) {
            //        intersection
            //          .map(1.0 / propFreq(_))
            //          .sum / union.map(1.0 / propFreq(_)).sum
            0.0
          } else {
            intersection.size.toDouble / (union.size + 1)
          }
        } else {
          0.0
        }
      } else {
        0.0
      }
    }
  }

  private def calcAvgItemSim[K](
      map: mutable.Map[K, mutable.ArrayBuffer[ItemId]],
      key: K,
      item: Item
  ) = {
    if (map contains key) {
      val propItem        = itemProperties(item.itemId)
      val prvInteractions = map(key).distinct
      if (propItem.nonEmpty & prvInteractions.nonEmpty) {
        val sim = prvInteractions.map(itemProperties).map { prvItem =>
          prvItem.intersect(propItem).size.toDouble / (prvItem
            .union(propItem)
            .size + 1)
        }
        sim.sum / sim.size
      } else {
        0.0
      }
    } else {
      0.0
    }
  }

  private def updateOrCreateSetMap[K, V](
      map: mutable.Map[K, mutable.ArrayBuffer[V]],
      key: K,
      value: V
  ): Unit = {
    if (!map.contains(key)) {
      map(key) = mutable.ArrayBuffer[V]()
    }
    map(key).append(value)
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
