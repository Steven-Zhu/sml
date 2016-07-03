package com.steven.siml

import scala.collection.mutable


class NaiveBayes {

    var probY: mutable.HashMap[String, Double] = new mutable.HashMap[String, Double]()
    var probXGivenY: mutable.HashMap[String, mutable.HashMap[String, Double]] =
        new mutable.HashMap[String, mutable.HashMap[String, Double]]()
    var EVENT_MODEL = false
    val X_ATTRIBUTE = new mutable.HashSet[String]()

    def train(X: Seq[Seq[(String, Int)]], Y: Seq[String], lambda: Double = 0.03) = {
        probY.clear()
        probXGivenY.clear()

        if (X.size != Y.length) throw new IllegalArgumentException();

        val countY = new mutable.HashMap[String, Int]
        val contXGivenY = new mutable.HashMap[String, mutable.HashMap[String, Int]]

        for (i <- Range(0, Y.length)) {
            val y = Y(i)
            countY.put(y, countY.getOrElse(y, 0) + 1)
            if (!contXGivenY.contains(y)) contXGivenY.put(y, new mutable.HashMap[String, Int]())

            val xCount = contXGivenY.get(y).get

            X(i).foreach(entry => {
                xCount.put(entry._1, xCount.getOrElse(entry._1, 0) + 1);
                X_ATTRIBUTE.add(entry._1)
            })

        }

        val yLen = countY.values.reduce(_ + _)
        countY.map(entry => probY.put(new String(entry._1), (entry._2.toDouble + lambda) / (yLen + countY.size * lambda)))

        contXGivenY.map(entry => probXGivenY.put(entry._1, {
            val xMap = new mutable.HashMap[String, Double]();
            entry._2.map(entryX => xMap.put(entryX._1,
                (entryX._2 + lambda) / (countY.get(entry._1).get + X_ATTRIBUTE.size * lambda)));

            xMap
        }))

        probXGivenY.foreach(entry => {
            val yLen = countY.get(entry._1).get
            val defaultProb = lambda / (yLen + X_ATTRIBUTE.size * lambda)
            X_ATTRIBUTE.foreach(x => entry._2.put(x,
                entry._2.getOrElse(x, defaultProb)))
        })
    }

    def test(X: Seq[Seq[(String, Int)]], Y: Seq[String]) = {
        if (X.size != Y.length) throw new IllegalArgumentException();

        var correct = 0

        for (i <- Range(0, Y.length)) {
            val probYGivenX: mutable.Map[String, Double] = new mutable.HashMap[String, Double]
            for (y <- probY.keys) {
                val h = {
                    val tmp = X(i).map { case (k: String, v: Int) => {
                        probXGivenY.get(y).get.get(k)
                    }
                    }
                        .filter(_.isDefined).map(_.get)
                    tmp.reduce(_ * _)
                }

                probYGivenX.put(y, h * probY.get(y).get)
            }
            if (probYGivenX.toSeq.sortWith(_._2 > _._2)(0)._1 == Y(i)) correct += 1
        }

        val acc = correct.toDouble / Y.length
        //    println("naive bayes acc: " + acc)
        acc
    }

}

object NaiveBayes {
    def main(args: Array[String]) {
        val trainPath = "20_newsgroups.train"
        val testPath = "20_newsgroups.test"
        val train = DataImporter.readsparse(Common.readResourceAsLines(trainPath))
        val Xtrain = train._1
        val Ytrain = train._2

        val test = DataImporter.readsparse(Common.readResourceAsLines(testPath))
        val Xtest = test._1
        val Ytest = test._2

        val model = new NaiveBayes()
        for (i <- Range(0, 3)) {
            model.train(Xtrain, Ytrain, 0.001 * i)
            val accTrain = model.test(Xtrain, Ytrain)
            val accTest = model.test(Xtest, Ytest)
            println("\nlambda: " + 0.001 * i)
            println("accTrain: " + accTrain)
            println("accTest: " + accTest)
        }
    }
}
