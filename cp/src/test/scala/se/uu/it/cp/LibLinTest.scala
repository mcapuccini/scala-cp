package se.uu.it.cp

import scala.io.Source
import scala.util.Random

import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner

import de.bwaldvogel.liblinear.Feature
import de.bwaldvogel.liblinear.FeatureNode
import de.bwaldvogel.liblinear.Linear
import de.bwaldvogel.liblinear.Parameter
import de.bwaldvogel.liblinear.Problem
import de.bwaldvogel.liblinear.SolverType

@RunWith(classOf[JUnitRunner])
class LibLinTest extends FunSuite {

  test("Train an inductive classifier with LIBLINEAR") {
 
    // Define a LIBLINEAR data point
    case class LibLinPoint(features: Array[Feature], label: Double)

    // Define a LIBLINEAR underlying algorithm
    class LibLinAlg(val properTrainingSet: Seq[LibLinPoint])
        extends UnderlyingAlgorithm[LibLinPoint] {
      
      // First describe how to access LIBLINEAR data point structure 
      override def makeDataPoint(features: Seq[Double], label: Double) = {
        val libLinFeat = features.zipWithIndex.map {
          case (f, i) =>
            new FeatureNode(i + 1, f).asInstanceOf[Feature]
        }
        LibLinPoint(libLinFeat.toArray, label)
      }
      override def getDataPointFeatures(p: LibLinPoint) = p.features.map(_.getValue)
      override def getDataPointLabel(p: LibLinPoint) = p.label

      // Train a Logistic Regression model
      val lrModel = {
        val problem = new Problem()
        problem.l = properTrainingSet.length
        problem.n = properTrainingSet(0).features.length
        problem.x = properTrainingSet.map(_.features).toArray
        problem.y = properTrainingSet.map(_.label).toArray
        val solver = SolverType.L2R_LR
        val parameter = new Parameter(solver, 1.0, 0.01)
        Linear.train(problem, parameter)
      }
      
      // Define nonconformity measure as probability of wrong prediction
      override def nonConformityMeasure(p: LibLinPoint) = {
        val estimates = Array.fill(2)(0.0)
        Linear.predictProbability(lrModel, p.features, estimates)
        estimates((p.label - 1).abs.toInt)
      }
      
    }
    
    // Load and parse dataset
    val dataPath = getClass.getResource("breast-cancer.data").getPath
    val dataset = Source.fromFile(dataPath).getLines
      .map { line =>
        val split = line.split(" ")
        val label = split(0).toDouble
        val features = split.drop(1).map { featureString =>
          val split = featureString.split(":")
          new FeatureNode(split(0).toInt,split(1).toDouble)
            .asInstanceOf[Feature]
        }
        LibLinPoint(features,label)
      }.toSeq
    
    // Split data
    Random.setSeed(11L)
    val (training,test) = Random.shuffle(dataset).splitAt(450)
    val (properTraining, calibrationSet) = Random.shuffle(training).splitAt(400)
    
    // Train an inductive conformal classifier
    val cp = ICP.trainClassifier(
      new LibLinAlg(properTraining), nOfClasses = 2, calibrationSet)

    // Make some predictions, and compute error fraction
    val significance = 0.05
    val nOfCorrect = test.count { p =>
      val pSet = cp.predict(p.features.map(_.getValue), significance)
      !pSet.contains(p.label)
    }
    val errorFract = nOfCorrect.toDouble / test.length

    // Error fraction should be at most significance 
    assert(errorFract <= significance)

  }

}