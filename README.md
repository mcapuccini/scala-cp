# Scala-CP

[![Build Status](https://travis-ci.org/mcapuccini/scala-cp.svg?branch=master)](https://travis-ci.org/mcapuccini/scala-cp)

Scala-CP is a Scala implementation of the Conformal Prediction (CP) framework, introduced by Vovk *et. al.* in the book Algorithmic Learning in a Random World. When assigning confidence to machine learning models, CP is a nice alternative to cross-validation. Instead of predicting a value for a certain feature vector, a conformal predictor outputs a prediction set/region that contains the correct prediction with probability *1-𝜺*, where *𝜺* is a user-defined significance level. The choose of the significance level will of course influence the size of the prediction set/region. In alternative, using CP one can predict object-specific p-values for unseen examples.

## Table of Contents
- [Getting started](#getting-started)
- [Documentation](#documentation)
  - [Scala-CP with Spark MLlib](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/SparkTest.scala)
  - [Scala-CP with LIBLINEAR](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/LibLinTest.scala)

## Getting started
Scala-CP can be used along with any Scala/Java machine learning library and algorithm. All you have to do is to add the Scala-CP dependency to your *pom.xml* file:

```xml
<dependencies>
	...
	<groupId>se.uu.it</groupId>
		<artifactId>cp</artifactId>
		<version>0.1.0</version>
	</dependency>
	...
</dependencies>
```

## Documentation
The API documentation is available at: https://mcapuccini.github.io/scala-cp/scaladocs/. For some usage examples you refer to the unit test:

  - [Scala-CP with Spark MLlib](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/SparkTest.scala)
  - [Scala-CP with LIBLINEAR](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/LibLinTest.scala)
