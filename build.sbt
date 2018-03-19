// Your sbt build file. Guides on how to write one can be found at
// http://www.scala-sbt.org/0.13/docs/index.html

spName := "com.dmeng/xgboostexample"

organization := "com.dmeng"

version := "0.1.0"

licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

scalaVersion := "2.11.8"

crossScalaVersions := Seq("2.10.5", "2.11.8")

sparkVersion := "2.1.0"

sparkComponents ++= Seq("mllib", "sql", "core", "graphx", "streaming")

spAppendScalaVersion := true

spIncludeMaven := true

spIgnoreProvided := true

test in assembly := {}

val testSparkVersion = settingKey[String]("The version of Spark to test against.")

testSparkVersion := sys.props.getOrElse("spark.testVersion", sparkVersion.value)

// Can't parallelly execute in test
parallelExecution in Test := false

fork in Test := true

javaOptions ++= Seq("-Xmx2G", "-XX:MaxPermSize=256m")

libraryDependencies ++= Seq(
		"junit" % "junit" % "4.12",
		"org.scalatest" %% "scalatest" % "2.2.6" % "test",
		"com.databricks" % "spark-csv_2.11" % "1.4.0",
        "com.github.scopt" % "scopt_2.10" % "3.3.0"
)

resolvers ++= Seq(
		"mapr-repo" at "http://repository.mapr.com/maven"
)
