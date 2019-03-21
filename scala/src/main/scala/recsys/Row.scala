package recsys

import recsys.Types.{ItemId, Price, UserId}

case class Row(
    actionType: String,
    userId: UserId,
    sessionId: String,
    timestamp: Integer,
    referenceItem: ItemId,
    referenceOther: String,
    platform: String,
    city: String,
    device: String,
    currentFilters: String,
    src: String,
    isTest: Boolean,
    impressions: Array[ItemId] = null,
    prices: Array[Price] = null,
    items: Array[Item] = null
)
