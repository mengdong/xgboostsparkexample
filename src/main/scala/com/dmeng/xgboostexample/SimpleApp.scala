package com.dmeng.xgboostexample

import scopt.OptionParser
import org.apache.log4j.{Level, Logger}

import org.apache.spark.sql.SparkSession
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostEstimator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator}


object SimpleApp extends Serializable{
    case class runParams(
        param1: String = null
    )

    def main(args: Array[String]): Unit = {

        Logger.getLogger("org").setLevel(Level.WARN)
        Logger.getLogger("akka").setLevel(Level.WARN)

        // spark.sparkContext.getConf.registerKryoClasses(Array(classOf[class1], classOf[class2]))
        val spark = SparkSession
            .builder()
            .appName("ExampleApp")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .enableHiveSupport()
            .getOrCreate()

        val defaultParam = new runParams()
        val parser = new OptionParser[runParams](this.getClass.getSimpleName) {
            head(s"${this.getClass.getSimpleName}: Run simple app")
            opt[String]("param1")
                .text("sample_param1")
                .action((x, c) => c.copy(param1 = x))
                .required()
        }
        parser.parse(args, defaultParam).map { params =>
            run(spark, params.param1)
        } getOrElse {
            sys.exit(1)
        }
        //sc.stop()
    }

    def run(spark: SparkSession, param1: String): Unit = {
        // val param1 = "sample content of param1"
        printf(s"Run the regression model with XGBoost: ......")
        val training = spark.read.format("libsvm")
            .load("data/sample_linear_regression_data.txt")
        val paramMap = List(
            "eta" -> 0.02,
            "max_depth" -> 8,
            "objective" -> "reg:linear").toMap
        val xgb = new XGBoostEstimator(paramMap)
        val pipeline = new Pipeline()
            .setStages(Array(xgb))
        val paramGrid = new ParamGridBuilder()
            .addGrid(xgb.minChildWeight, Array(1.0, 5.0))
            .build()
        // cv
        val evaluator = new RegressionEvaluator()
            .setMetricName("rmse")
        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(2)  // Use 3+ in practice

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(training)
        val predictions = cvModel.transform(training)
        val rmse = evaluator.evaluate(predictions)
        printf(s"RMSE is : $rmse")
        predictions.show

        printf(s"Run the classification model with XGBoost")
        val paramMap2 = List(
            "eta" -> 0.02,
            "max_depth" -> 8,
            "objective" -> "binary:logistic").toMap
        val training2 = spark.read.format("libsvm")
            .load("data/sample_libsvm_data.txt")
        val xgb2 = new XGBoostEstimator(paramMap2)
        val pipeline2 = new Pipeline()
            .setStages(Array(xgb2))
        val paramGrid2 = new ParamGridBuilder()
            .addGrid(xgb2.minChildWeight, Array(1.0, 5.0))
            .build()
        // cv
        val evaluator2 = new BinaryClassificationEvaluator()
            .setRawPredictionCol("probabilities")
        val cv2 = new CrossValidator()
            .setEstimator(pipeline2)
            .setEvaluator(evaluator2)
            .setEstimatorParamMaps(paramGrid2)
            .setNumFolds(2)  // Use 3+ in practice
        // Run cross-validation, and choose the best set of parameters.
        val cvModel2 = cv2.fit(training2)
        val predictions2 = cvModel2.transform(training2)
        val auc = evaluator2.evaluate(predictions2)
        printf(s"AUC is : $auc")
        predictions2.show

        val training3 = spark.read.format("libsvm")
            .load("data/sample_multiclass_classification_data.txt")
        printf(s"Run the classification model with XGBoost")
        val paramMap3 = List(
            "eta" -> 0.02,
            "max_depth" -> 8,
            "num_class" -> 3,
            "objective" -> "multi:softmax").toMap
        val xgb3 = new XGBoostEstimator(paramMap3)
        val pipeline3 = new Pipeline()
            .setStages(Array(xgb3))
        val xgbModel = pipeline3.fit(training3)

        val paramGrid3 = new ParamGridBuilder()
            .addGrid(xgb3.minChildWeight, Array(1.0, 5.0))
            .build()
        // cv
        val evaluator3 = new MulticlassClassificationEvaluator()
            .setPredictionCol("prediction")
            .setLabelCol("label")
        val cv3 = new CrossValidator()
            .setEstimator(pipeline3)
            .setEvaluator(evaluator3)
            .setEstimatorParamMaps(paramGrid3)
            .setNumFolds(2)  // Use 3+ in practice
        // Run cross-validation, and choose the best set of parameters.
        val cvModel3 = cv3.fit(training3)
        val predictions3 = cvModel3.transform(training3)
        val auc2 = evaluator3.evaluate(predictions3)
        printf(s"AUC is : $auc2")
        predictions3.show


    }
}

        
