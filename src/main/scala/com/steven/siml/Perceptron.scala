package com.steven.siml
import breeze.linalg.{DenseMatrix=>matrix, DenseVector=>vector}

import Common._

class Perceptron extends Model {
    var weight: vector[Double] = null
    var lRate: Double = 0.3
    var maxIter = 10
    var maxUpdate = 10
    var W:vector[Double] = null //n+1 dimension

    override def train(X: matrix[Double], Y: vector[Double], Weight: vector[Double]):
    (Model, vector[Double], Double) = {
        val range = Range(0, X.rows)

        val X0 = matrix.ones[Double](X.rows, 1)
        val Xtrain = matrix.horzcat(X, X0)
        val w = vector.zeros[Double](X.cols + 1)

        var indices:IndexedSeq[Int] = Range(0, X.rows)
        var updateTimes = 0

        var errNum = 1 //make it pass the first condision decision
        for (i <- Range(0, maxIter)) {

            if(errNum > 0 ) {
                errNum = 0 //initialized
                indices = scala.util.Random.shuffle[Int, IndexedSeq](indices)
                for (j <- indices) {
                    if (math.signum(Xtrain(j, ::) * w) != Y(j) && updateTimes < maxUpdate) {
                        errNum += 1
                        w :+= lRate * Xtrain(j, ::).t * Y(j) * Weight(j)
                        updateTimes += 1
                    }
                }
//                printf("Iter Times: %d \ntraining loss: %f\n", i, errNum.toDouble / X.rows)
            }
        }
        this.W = w
        val Ypred = classify(X)
        (this, Ypred, loss(Ypred, Y))
    }

    override def classify(X: matrix[Double]): vector[Double] = {
        val X0 = matrix.ones[Double](X.rows, 1)
        val Xtrain = matrix.horzcat(X, X0)
        val Ypred = sign(Xtrain * this.W)
        Ypred
    }
}

object Perceptron {
    def main(args: Array[String]) {
        val trainPath = "heart_scale.train"
        val testPath = "heart_scale.test"
        val train = DataImporter.readDense(Common.readResourceAsLines(trainPath))
        val Xtrain = train._1
        val Ytrain = train._2.toDenseVector

        val test = DataImporter.readDense(Common.readResourceAsLines(testPath))
        val Xtest = test._1
        val Ytest = test._2.toDenseVector

        val pcep = new Perceptron
        pcep.lRate = 0.1
        pcep.maxUpdate = 1000
        pcep.maxIter = 1000

        val K = pcep.train(Xtrain, Ytrain, vector.ones[Double](Xtrain.rows))
        printf("finished\ntraining loss: %f\n", K._3)

        val Ypred = K._1.classify(Xtest)
        printf("test loss: %f", loss(Ytest, Ypred))
    }
}
