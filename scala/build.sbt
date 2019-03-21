name := "scala"

version := "0.1"

scalaVersion := "2.12.8"

// https://mvnrepository.com/artifact/org.apache.flink/flink-clients
libraryDependencies += "org.apache.flink" %% "flink-scala"      % "1.7.2"
libraryDependencies += "org.apache.flink" %% "flink-clients"    % "1.7.2"
libraryDependencies += "com.univocity"    % "univocity-parsers" % "2.8.1"
libraryDependencies += "me.tongfei"       % "progressbar"       % "0.7.2"
