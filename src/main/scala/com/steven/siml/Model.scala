package com.steven.siml

import breeze.linalg.{DenseMatrix => matrix, DenseVector => vector}


trait Model {

    def train(X:matrix[Double], Y:vector[Double], Weight:vector[Double]): (Model, vector[Double], Double)

    def classify(X:matrix[Double]):vector[Double]

}
