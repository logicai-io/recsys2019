package recsys

import scala.util.Try

object Utils {
  def isAllDigits(x: String): Boolean =
    Try(x forall Character.isDigit).getOrElse(false)
}
