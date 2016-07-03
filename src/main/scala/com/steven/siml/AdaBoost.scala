package com.steven.siml

import breeze.linalg.{sum, DenseMatrix => matrix, DenseVector => vector}
import com.steven.siml.Common._

import scala.collection.mutable
import scala.reflect.ClassTag


class AdaBoost[T <: Model] {
    var loop = 5
    var models:Array[T] = null
    var alphas:Array[Double] = null

    def train(X: matrix[Double], Y: vector[Double], clazz: Class[T])(implicit m: ClassTag[T]) = {
        var weight = vector.tabulate(X.rows) {i => 1/ X.rows.toDouble}
        models = new Array[T](loop)
        alphas = new Array[Double](loop)
        for (i <- Range(0, loop)) {
            //(model, YPred, loss)
            val model = clazz.newInstance()
            val trainRes = model.train(X, Y, weight)
            models(i) = trainRes._1.asInstanceOf[T]
            val yPred: vector[Double] = trainRes._2
            val loss = trainRes._3

            println("loss: " + loss)

            alphas(i) = 0.5 * math.log((1.0 - loss) / loss)
            if(alphas(i) < 0 ) {alphas(i) = 0}
            else {
                val weightLocal = bitMultuply(weight, exp(-alphas(i) * bitMultuply(Y, yPred)))
                weight = weightLocal / sum(weightLocal)
                println("weight update: sum=" + sum(weight))
            }
        }
        println("alphas: ")
        alphas.foreach(println)
    }

    def test(X:matrix[Double]):vector[Double] = {
        val Ypred = new Array[Double](X.rows)

        val ress = models.map(model => model.classify(X))

        val resMap = new Array[mutable.HashMap[Double, Double]](X.rows)
        for (iRow <- resMap.indices) {
            val predMap = new mutable.HashMap[Double, Double]()
            for ( iPred <- ress.indices) {
                //its classification, use + alphas(iPred) to determine which class is good
                predMap.put(ress(iPred)(iRow), predMap.getOrElse(ress(iPred)(iRow), 0.0) + 1*alphas(iPred) )
            }

            var maxIndex = 0.0
            var maxValue = 0.0
            for ( k <- predMap.keys) {
                if (predMap.get(k).get > maxValue) {
                    maxValue = predMap.get(k).get
                    maxIndex = k
                }
            }
            Ypred(iRow) = maxIndex
        }
        new vector(Ypred)
    }
}
