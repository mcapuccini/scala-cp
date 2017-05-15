package se.uu.it.cp

import org.apache.log4j.Logger

/**
 * Inductive conformal classification model.
 *
 *  @constructor
 *  @param alg an underlying algorithm
 *  @param alphas nonconformity scores computed from a calibration set
 *  	(which is unseen to the underlying algorithm)
 */
class InductiveClassifier[
      A <: UnderlyingAlgorithm[Data, DataPoint], 
      Data <: Any, 
      DataPoint <: Any](
    val alg: A,
    val alphas: Seq[Seq[Double]]) extends Serializable {

  @transient private lazy val log = Logger.getLogger(getClass.getName)

  /**
   * Given a feature sequence returns a p-value for each class.
   *
   *  @param alg an underlying algorithm
   *  @param alphas nonconformity scores computed from a calibration set
   *  	(which is unseen to the underlying algorithm)
   *  @return a sequence of p-values (one for each class)
   */
  def mondrianPv(features: Seq[Double]) = {
    //Compute mondrian p-values
    (0 to alphas.length - 1).map { i =>
      //compute non-conformity for new example
      val alphaN = alg.nonConformityMeasure(alg.makeDataPoint(features, i))
      //compute p-value
      (alphas(i).count(_ >= alphaN) + 1).toDouble /
        (alphas(i).length.toDouble + 1)
    }
  }

  /**
   * Computes a prediction set for a feature sequence.
   *
   *  @param features a feature sequence
   *  @param significance a significance level bigger than 0 and smaller than 1
   *  @return prediction set
   */
  def predict(features: Seq[Double], significance: Double) = {
    //Validate input
    require(significance > 0 && significance < 1, s"significance $significance is not in (0,1)")
    alphas.foreach { a =>
      if (a.length < 1 / significance - 1) {
        log.warn(s"too few calibration samples (${a.length}) for significance $significance")
      }
    }
    //Compute prediction set
    mondrianPv(features).zipWithIndex.map {
      case (pVal, c) =>
        if (pVal > significance) {
          Set(c.toDouble)
        } else {
          Set[Double]()
        }
    }.reduce(_ ++ _)
  }

}
