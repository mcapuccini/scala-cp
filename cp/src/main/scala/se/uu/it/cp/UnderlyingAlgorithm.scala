package se.uu.it.cp

/**
 * Represents a labeled data point.
 *
 *  @constructor 
 *  @param features a sequence of features
 *  @param label the data point label
 */
case class DataPoint(
  features: Seq[Double],
  label: Double)

/**
 * Defines an underlying algorithm to be used when training a conformal predictor.
 *
 *  @constructor 
 *  @param predictor a lambda function that given a sequence of features predicts a label
 */
abstract class UnderlyingAlgorithm(
    val predictor: (Seq[Double] => Double)) extends Serializable {
  
  /**
   * Defines a nonconformity measure that given an unseen sample returns a nonconformity score.
   *
   *  @param newSample an unseen sample
   *  @return nonconformity score for newSample
   */
  def nonConformityMeasure(newSample: DataPoint): Double
  
}