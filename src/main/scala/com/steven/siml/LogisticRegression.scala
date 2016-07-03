package com.steven.siml

import breeze.linalg.{sum, DenseMatrix => matrix}
import com.steven.siml.Common._


class LogisticRegression {
    var W: matrix[Double] = null

    def train(X: matrix[Double], Y: matrix[Double]): Unit = {
        val x0 = matrix.ones[Double](X.rows, 1)
        val X1 = matrix.horzcat(X, x0)
        val w = matrix.zeros[Double](X.cols + 1, 1)
        W = GradientDescent.gd(cost, w, X1, Y)
    }

    def cost(W: matrix[Double], X: matrix[Double], Y: matrix[Double])
    : (matrix[Double], matrix[Double]) = {
        val S = sigmoid(X * W)
        val funcVal = -Y.t * log(S) - (1.0 - Y).t * (1.0 - log(S))
        val funcGradient = X.t * (S - Y)
        (funcVal, funcGradient)
    }

    def test(X: matrix[Double], Y: matrix[Double]): Double = {
        val x0 = matrix.ones[Double](X.rows, 1)
        val X1 = matrix.horzcat(X, x0)

        val YPred = sign(X1 * W).map(xi => if (xi == 1.0) 1.0; else 0.0)

        val P = (Y :== YPred).map(xi => if (xi) 1; else 0)
        sum(P).toDouble / Y.rows
    }

}

object LogisticRegression {

    def main(args: Array[String]) {
        val trainPath = "heart_scale.train"
        val testPath = "heart_scale.test"
        val train = DataImporter.readDense(Common.readResourceAsLines(trainPath))
        val Xtrain = train._1
        val Ytrain = train._2.map(xi => if (xi == -1.0) 0; else xi)

        val test = DataImporter.readDense(Common.readResourceAsLines(testPath))
        val Xtest = test._1
        val Ytest = test._2.map(xi => if (xi == -1.0) 0; else xi)

        for (j <- Range(1, 10)) {

            GradientDescent.maxIter = 20 * j
            GradientDescent.step = 0.01
            val LR = new LogisticRegression()
            LR.train(Xtrain, Ytrain)
            val accTrain = LR.test(Xtrain, Ytrain)
            val accTest = LR.test(Xtest, Ytest)
            println("\nmaxiter: " + GradientDescent.maxIter + " step: " + GradientDescent.step)
            println("accTrain: " + accTrain)
            println("accTest: " + accTest)
        }
    }
}
