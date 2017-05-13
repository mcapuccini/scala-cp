package se.uu.it.cp

import scala.util.Random

import org.apache.log4j.BasicConfigurator
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

case class DataPoint(features: Seq[Double], label: Double)

object ICPTest {

  def generateFourClassesData(numSamples: Int) = {
    Seq.fill(numSamples)((math.random, math.random)).map {
      case (x0, x1) =>
        val label = if (x0 < 0.5 && x1 < 0.5) {
          0.0
        } else if (x0 < 0.5) {
          1.0
        } else if (x1 < 0.5) {
          2.0
        } else {
          3.0
        }
        DataPoint(Seq(x0, x1), label)
    }
  }

  def randomSplitAt(data: Seq[DataPoint], n: Int) = {
    // Balanced split
    val (dataLeft0, dataRight0) = Random.shuffle(data.filter(_.label == 0.0)).splitAt(n / 4)
    val (dataLeft1, dataRight1) = Random.shuffle(data.filter(_.label == 1.0)).splitAt(n / 4)
    val (dataLeft2, dataRight2) = Random.shuffle(data.filter(_.label == 2.0)).splitAt(n / 4)
    val (dataLeft3, dataRight3) = Random.shuffle(data.filter(_.label == 3.0)).splitAt(n / 4)
    val dataLeft = Random.shuffle(dataLeft0 ++ dataLeft1 ++ dataLeft2 ++ dataLeft3)
    val dataRight = Random.shuffle(dataRight0 ++ dataRight1 ++ dataRight2 ++ dataRight3)
    (dataLeft, dataRight)
  }

}

@RunWith(classOf[JUnitRunner])
class ICPTest extends FunSuite {

  BasicConfigurator.configure // configure log4j

  // Define 5NN underlying algorithm
  class FiveNN(val properTrainingSet: Seq[DataPoint])
      extends UnderlyingAlgorithm(properTrainingSet) {
    def makeDataPoint(features: Seq[Double], label: Double) = DataPoint(features, label)
    def getDataPointFeatures(dataPoint: DataPoint) = dataPoint.features
    def getDataPointLabel(dataPoint: DataPoint) = dataPoint.label
    def trainingProcedure(trainingSet: Seq[DataPoint]) = (features: Seq[Double]) => {
      val trainByDistance = trainingSet.sortBy { point =>
        val squaredDist = point.features.zip(features).map {
          case (q, p) => math.pow(q - p, 2)
        }.reduce(_ + _)
        math.sqrt(squaredDist)
      }
      trainByDistance.take(5)
        .groupBy(_.label)
        .maxBy(_._2.size)
        ._1
    }
    def nonConformityMeasure(newSample: DataPoint) = {
      // Fraction of different closest neighbours
      val trainByDistance = properTrainingSet.sortBy { point =>
        val squaredDist = point.features.zip(newSample.features).map {
          case (q, p) => math.pow(q - p, 2)
        }.reduce(_ + _)
        math.sqrt(squaredDist)
      }
      val nOfUnconformal = trainByDistance.take(5).count(_.label != newSample.label)
      nOfUnconformal.toDouble / 5
    }
  }

  test("ICP classification: error should be lower than significance on average") {

    // Set a significance level
    val significance = 0.35
    //Generate some sample data
    val sampleData = ICPTest.generateFourClassesData(100)

    // Train and compute error fractions for 100 calibration set choices 
    val errFracts = (0 to 100).map { _ =>
      // Split proper training, calibration and test
      val (training, test) = ICPTest.randomSplitAt(sampleData, 80)
      val (properTraining, calibration) = ICPTest.randomSplitAt(training, 60)
      // Train ICP
      val alg = new FiveNN(properTraining)
      val model = ICP.trainClassifier(alg, nOfClasses = 4, calibration)
      //compute and return error fraction
      val errors = test.count { p =>
        val pSet = model.predict(p.features, significance)
        !pSet.contains(p.label)
      }
      errors.toDouble / test.length
    }

    // Error should be at most equal to significance on average
    val avgError = errFracts.sum / errFracts.length
    assert(avgError <= significance)

  }

}