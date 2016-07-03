package com.steven.siml

import breeze.linalg.{sum, DenseMatrix => matrix}


class SVM_SMO {
    var W: matrix[Double] = null
    var b: Double = 0
    var kernal: (matrix[Double], matrix[Double]) => matrix[Double] = linear
    var maxPasses = 6
    var tolerance = 1e-3
    var C: Double = 1

    var KCache: matrix[Double] = null

    def linear(xi: matrix[Double], xj: matrix[Double]) = {
        //输入的X看做是行向量构成的
        xi * xj.t
    }

    def calGxi(alphas: matrix[Double], b: Double, Y: matrix[Double], i: Int) = {
        val tmp: matrix[Double] = alphas.t * Common.bitMultuply(Y, KCache(::, i).toDenseMatrix.t) + b;
        tmp(0, 0)
    }

    def train(X: matrix[Double], Y: matrix[Double]): Unit = {
        //输入的X看做是行向量构成的， Y则是列向量
        val alphas = matrix.zeros[Double](X.rows, 1)
        var b = 0.0
        KCache = kernal(X, X)

        var passes = 0
        while (passes < maxPasses) {
            var numAlphaChanged = 0
            for (i <- Range(0, X.rows)) {
                val gxi = calGxi(alphas, b, Y, i)
                val yi = Y(i, 0)
                val Ei = gxi - yi
                val alphaI = alphas(i, 0)
                if ((alphaI < C && yi * Ei < -tolerance) || (alphaI > 0 && yi * Ei > tolerance)) {
                    //选择第一个变量
                    var j = 0
                    while ( {
                        j = (math.random * (X.rows)).toInt;
                        j == i
                    }) {} //选择第二个变量
                    val gxj = calGxi(alphas, b, Y, j)
                    val yj = Y(j, 0)
                    val Ej = gxj - yj
                    val alphaJ = alphas(j, 0)
                    var L: Double = 0
                    var H: Double = 0
                    if (yi != yj) {
                        L = math.max(0, alphaJ - alphaI)
                        H = math.min(C, C + alphaJ - alphaI)
                    } else {
                        L = math.max(0, alphaJ + alphaI - C)
                        H = math.min(C, alphaJ + alphaI)
                    }
                    val eta = KCache(i, i) + KCache(j, j) - 2 * KCache(i, j)
                    if (eta > 0) {
                        val alphasJnew = {
                            val alphaJLocal = alphaJ + (yj * (Ei - Ej)) / eta
                            if (alphaJLocal > H) H
                            else if (alphaJLocal >= L) alphaJLocal
                            else L
                        }
                        if (alphasJnew - alphaJ > tolerance) {
                            alphas(j, 0) = alphasJnew
                            alphas(i, 0) = alphas(i, 0) + yi * yj * (alphaJ - alphas(j, 0))

                            val bi = -Ei - yi * KCache(i, i) * (alphas(i, 0) - alphaI) - yj * KCache(j, i) * (alphas(j, 0) - alphaJ) + b
                            val bj = -Ej - yi * KCache(i, j) * (alphas(i, 0) - alphaI) - yj * KCache(j, j) * (alphas(j, 0) - alphaJ) + b
                            if (0 < alphas(i, 0) && alphas(i, 0) < C) b = bi
                            else if (0 < alphas(j, 0) && alphas(j, 0) < C) b = bj
                            else b = (bi + bj) / 2

                            numAlphaChanged += 1
                        }
                    }
                }
            }
            if (numAlphaChanged == 0) passes += 1
            else passes = 0
        }

        this.b = b
        this.W = X.t * Common.bitMultuply(alphas, Y)
    }

    def predict(X: matrix[Double]): matrix[Double] = {
        X * this.W + b
    }

    def test(X: matrix[Double], Y: matrix[Double]): Double = {
        val px = predict(X)
        val yPred = Common.sign(px)

        val yTrue = matrix.tabulate(Y.rows, 1) { case (i, j) => if (Y(i, 0) == yPred(i, 0)) 1.0; else 0.0 }
        sum(yTrue) / Y.rows
    }

}

object SVM_SMO {
    def main(args: Array[String]) {
        val trainPath = "heart_scale.train"
        val testPath = "heart_scale.test"
        val train = DataImporter.readDense(Common.readResourceAsLines(trainPath))
        val Xtrain = train._1
        val Ytrain = train._2

        val test = DataImporter.readDense(Common.readResourceAsLines(testPath))
        val Xtest = test._1
        val Ytest = test._2

        val model = new SVM_SMO()

        for (i: Double <- Seq(0.5, 1.0, 2, 4)) {
            model.train(Xtrain, Ytrain)
            model.C = i
            println("C: " + i)
            println("accTrain: " + model.test(Xtrain, Ytrain) * 100 + "%")
            println("accTest: " + model.test(Xtest, Ytest) * 100 + "%")
        }
    }
}
