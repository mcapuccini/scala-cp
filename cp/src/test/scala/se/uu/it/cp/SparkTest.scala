package se.uu.it.cp

import org.scalatest.FunSuite
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.SVMWithSGD
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

// Define a MLlib SVM underlying algorithm
class SVM(trainingSet: RDD[LabeledPoint])
    extends UnderlyingAlgorithm[RDD[LabeledPoint], LabeledPoint](trainingSet) {
  override def makeDataPoint(features: Seq[Double], label: Double) =
    new LabeledPoint(label, Vectors.dense(features.toArray))
  override def getDataPointFeatures(lp: LabeledPoint) = lp.features.toArray
  override def getDataPointLabel(lp: LabeledPoint) = lp.label
  override def trainingProcedure(trainingSet: RDD[LabeledPoint]) = {
    // Train with SVMWithSGD
    val svmModel = SVMWithSGD.train(trainingSet, numIterations = 100)
    svmModel.clearThreshold // set to return distance from hyperplane
    (features: Seq[Double]) => svmModel.predict(Vectors.dense(features.toArray))
  }
  override def nonConformityMeasure(lp: LabeledPoint) = {
    if (lp.label == 1.0) {
      -predictor(lp.features.toArray)
    } else {
      predictor(lp.features.toArray)
    }
  }
}

@RunWith(classOf[JUnitRunner])
class SparkTest extends FunSuite {

  test("Train an inductive classifier with Apache Spark") {

    // Start SparkContext
    val conf = new SparkConf().setMaster("local[*]").setAppName("test")
    val sc = new SparkContext(conf)

    // Load a dataset
    val dataPath = getClass.getResource("breast-cancer.data").getPath
    val data = MLUtils.loadLibSVMFile(sc, dataPath)

    // Split data
    // Warning: calibration fraction should be lower for big datasets
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val Array(properTraining, calibrationSet) =
      training.randomSplit(Array(0.7, 0.3), seed = 11L)

    // Train an inductive conformal classifier
    val cp = ICP.trainClassifier[SVM, RDD[LabeledPoint], LabeledPoint](
      new SVM(properTraining.cache()), nOfClasses = 2, calibrationSet.collect)

    // Make some predictions, and compute error fraction
    val significance = 0.05
    val nOfCorrect = test.filter { lp =>
      val pSet = cp.predict(lp.features.toArray, significance)
      !pSet.contains(lp.label)
    }.count
    val errorFract = nOfCorrect.toDouble / test.count

    // Error fraction should be at most significance 
    assert(errorFract <= significance)

  }

}
