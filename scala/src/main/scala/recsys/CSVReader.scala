package recsys

import java.io.File

import com.univocity.parsers.csv.{CsvParser, CsvParserSettings}

object CSVReader {
  def getCSVReader(filePath: String) = {
    val settings = new CsvParserSettings()
    settings.setHeaderExtractionEnabled(true)
    val reader = new CsvParser(settings)
    val it     = reader.iterateRecords(new File(filePath), "UTF-8")
    it.iterator()
  }
}
