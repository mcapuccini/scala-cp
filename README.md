# Scala-CP

[![Build Status](https://travis-ci.org/mcapuccini/scala-cp.svg?branch=master)](https://travis-ci.org/mcapuccini/scala-cp)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/810ed0d38e6f47079eab3426f6bf6f95)](https://www.codacy.com/app/m-capuccini/scala-cp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mcapuccini/scala-cp&amp;utm_campaign=Badge_Grade)

Scala-CP is a Scala implementation of the Conformal Prediction (CP) framework, introduced by Vovk *et. al.* in the book Algorithmic Learning in a Random World. When assigning confidence to machine learning models, CP is a nice alternative to cross-validation. Instead of predicting a value for a certain feature vector, a conformal predictor outputs a prediction set/region that contains the correct prediction with probability *1-𝜺*, where *𝜺* is a user-defined significance level. The choose of the significance level will of course influence the size of the prediction set/region. In alternative, using CP one can predict object-specific p-values for unseen examples.

## Table of Contents
- [Getting started](#getting-started)
- [Documentation](#documentation)
  - [Scala-CP with Spark MLlib](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/SparkTest.scala)
  - [Scala-CP with LIBLINEAR](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/LibLinTest.scala)
- [List of publications](#list-of-publications)
- [Roadmap](#roadmap)

## Getting started
Scala-CP can be used along with any Scala/Java machine learning library and algorithm. All you have to do is to add the Scala-CP dependency to your *pom.xml* file:

```xml
<dependencies>
  ...
  <dependency>
    <groupId>se.uu.it</groupId>
    <artifactId>cp</artifactId>
    <version>0.1.0</version>
  </dependency>
  ...
</dependencies>
```

## Documentation
The API documentation is available at: https://mcapuccini.github.io/scala-cp/scaladocs/. For some usage examples please refer to the unit tests:

  - [Scala-CP with Spark MLlib](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/SparkTest.scala)
  - [Scala-CP with LIBLINEAR](https://github.com/mcapuccini/scala-cp/blob/master/cp/src/test/scala/se/uu/it/cp/LibLinTest.scala)
  
You can also refer to this Apache Zeppelin notebooks for more examples:

  - [ZeppelinHub: Scala-CP with Spark MLlib](https://www.zeppelinhub.com/viewer/notebooks/bm90ZTovL21jYXB1Y2NpbmkvY29uZm9ybWFsLXByZWRpY3Rpb24vYTYzMzJkOTcxZjYzNDBhZDg1NmQwMjRkNmE1NDliMzIvbm90ZS5qc29u)
  
## List of publications
- [M. Capuccini, L. Carlsson, U. Norinder and O. Spjuth, "Conformal Prediction in Spark: Large-Scale Machine Learning with Confidence," 2015 IEEE/ACM 2nd International Symposium on Big Data Computing (BDC), Limassol, 2015, pp. 61-67.](http://ieeexplore.ieee.org/document/7406330/)

## Roadmap

### Inductive Conformal Prediction
- [x] Classification 
- [ ] Regression

### Transductive Conformal Prediction
- [ ] Classification 
- [ ] Regression
