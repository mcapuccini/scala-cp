package se.uu.it.cp

object ICP {

  /**
   * Trains an inductive conformal classifier using the mondrian approach.
   *
   *  @param algorithm an underlying algorithm to be used
   *  @nOfClasses number of classes
   *  @calibrationSet calibration set which is unseen to the underlying algorithm
   *  	training procedure
   *  @return inductive conformal classifier
   */
  def trainClassifier[A <: UnderlyingAlgorithm](
    algorithm: A,
    nOfClasses: Int,
    calibrationSet: Seq[DataPoint]): InductiveClassifier[A] = {
    //Compute alphas for each class (mondrian approach)
    val alphas = (0 to nOfClasses - 1).map { i =>
      calibrationSet.filter(_.label == i) //filter current label
        .map(newSmpl => algorithm.nonConformityMeasure(newSmpl)) //compute alpha
    }
    //Return an InductiveClassifier
    new InductiveClassifier(algorithm, alphas)
  }

}