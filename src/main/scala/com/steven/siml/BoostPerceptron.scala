package com.steven.siml


object BoostPerceptron {

    def main(args: Array[String]) {
        val trainPath = "heart_scale.train"
        val testPath = "heart_scale.test"
        val train = DataImporter.readDense(Common.readResourceAsLines(trainPath))
        val Xtrain = train._1
        val Ytrain = train._2.toDenseVector

        val test = DataImporter.readDense(Common.readResourceAsLines(testPath))
        val Xtest = test._1
        val Ytest = test._2.toDenseVector

        val adaBoost = new AdaBoost[Perceptron]
        adaBoost.loop = 30
        adaBoost.train(Xtrain, Ytrain, classOf[Perceptron])

        println("adaboost test loss: ")
        println(Common.loss(Ytest, adaBoost.test(Xtest)))

    }

}
