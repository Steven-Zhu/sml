package com.steven.siml

import breeze.linalg.{DenseMatrix => matrix, DenseVector => vector}
import breeze.linalg._

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.tools.nsc.interpreter.InputStream


object Common {
    def sigmoid(Z: vector[Double]): vector[Double] = {
        Z.map(xi => 1 / (1 + math.exp(-xi)))
    }

    def sigmoid(Z: matrix[Double]): matrix[Double] = {
        Z.map(xi => 1 / (1 + math.exp(-xi)))
    }

    def log(Z: vector[Double]): vector[Double] = {
        Z.map(xi => math.log(xi))
    }

    def log(Z: matrix[Double]): matrix[Double] = {
        Z.map(xi => math.log(xi))
    }

    def exp(Z: matrix[Double]): matrix[Double] = {
        Z.map(xi => math.exp(xi))
    }

    def exp(Z: vector[Double]): vector[Double] = {
        Z.map(xi => math.exp(xi))
    }

    def loss(Y:vector[Double], Ytest:vector[Double]):Double = {
        val comp = vector.tabulate(Ytest.length) { case (i) => if (Y(i) == Ytest(i)) 1; else 0 }
        1 - (sum(comp).toDouble / comp.length)
    }

    def bitMultuply(x: matrix[Double], y: matrix[Double]): matrix[Double] = {
        if (x.rows != y.rows || x.cols != y.cols) throw new IllegalArgumentException()

        matrix.tabulate(x.rows, x.cols) { case (i, j) => x(i, j) * y(i, j) }
    }

    def bitMultuply(x: vector[Double], y: vector[Double]): vector[Double] = {
        if (x.length != y.length ) throw new IllegalArgumentException()

        vector.tabulate(x.length) { i => x(i) * y(i) }
    }

    def sign(Z: matrix[Double]): matrix[Double] = {
        Z.map(xi => math.signum(xi))
    }

    def sign(Z: vector[Double]): vector[Double] = {
        Z.map(xi => math.signum(xi))
    }

    def readResource(path: String): InputStream = {
        this.getClass.getClassLoader.getResourceAsStream(path)
    }

    def readResourceAsLines(path: String): Iterator[String] = {
        Source.fromInputStream(this.getClass.getClassLoader.getResourceAsStream(path)).getLines()
    }

    /**
      * ((Xtrain, Ytrain),(Xtest, Ytest))
      * @param X
      * @param Y
      * @param fold
      * @return
      */
    def sampling(X:matrix[Double], Y:vector[Double], fold:Int):
    ((matrix[Double], vector[Double]), (matrix[Double], vector[Double])) = {
        if (X.rows != Y.length) throw new IllegalArgumentException("X.rows doesnot confirm to Y.length")

        val testSamples = new ListBuffer[Int]
        for ( i <- Range(0, X.rows/fold)) {
            testSamples.append(i * fold + (math.random * fold).toInt)
        }

        val samples = Range(0, X.rows).toSet -- testSamples.toSet
        ((X(samples.toIndexedSeq, ::).toDenseMatrix, Y(samples.toIndexedSeq).toDenseVector),
            (X(testSamples.toIndexedSeq, ::).toDenseMatrix, Y(testSamples.toIndexedSeq).toDenseVector))
    }

    def main(args: Array[String]) {

        println(bitMultuply(matrix(1.0, 2.0, 3.0, 4.0), matrix(2.0, 1.0, 3.0, 1.0)))
    }

}
