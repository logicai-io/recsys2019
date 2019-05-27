package recsys
import scala.io.Source
import scala.util.parsing.json._

case class ClickWithTS(timestamp: Int, index: Int)

object GeneticAlgorithm {
  def main(args: Array[String]): Unit = {
    val filename = "../src/recsys/genetic_programming/click_indices.ndjson"

    val sessions = Source.fromFile(filename).getLines.take(10).map { line =>
      var parsed = JSON.parseFull(line)
      var parsed2 = parsed match {
        case Some(e) => e
        case None => null
      }
      val userSessions = parsed2.asInstanceOf[List[List[List[Double]]]]

      userSessions.map {
        sessions => sessions.map {
          clicks => ClickWithTS(clicks(0).toInt, clicks(1).toInt)
        }.toVector
      }.toVector
    }.toVector

    sessions.foreach(println(_))
  }
}
