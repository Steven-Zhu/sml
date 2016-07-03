package com.steven.siml

import breeze.linalg._

object GradientDescent {

  var step:Double = 0.01
  var maxIter: Int = 50
  var tolerance : Double = 0.1

  def gd(func: (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) => (DenseMatrix[Double], DenseMatrix[Double]),
         W : DenseMatrix[Double],
         X: DenseMatrix[Double], Y:DenseMatrix[Double]): DenseMatrix[Double] = {

    var (funcVal , funcGradientVal) = func(W, X, Y)
    var iter = 0
    val M = funcGradientVal.rows
    val K = DenseMatrix.ones[Double](M, 1)
    while (iter < maxIter && {val h = (K.t * funcGradientVal); math.abs(h(0, 0)) > tolerance}){
      W -= step :* funcGradientVal
      val (f, fG) = func(W,  X, Y)
      funcVal = f
      funcGradientVal = fG
      iter += 1
    }
    W
  }

}
