package com.steven.siml

import java.io.File

import breeze.linalg.{sum, DenseMatrix => matrix, DenseVector => vector}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import Common._


class DecisionTree {
    var X: matrix[Double] = null
    var Y: vector[Double] = null
    var giniThreshold = 0.01
    var samplesThreshold = 4
    var Attributes: Seq[Seq[Double]] = null
    var dt: Node = null
    var gama = 1e-1
    var maxDepth = 15

    def train(X: matrix[Double], Y: vector[Double]) = {
        this.X = X
        this.Y = Y
        val XVals: ListBuffer[Seq[Double]] = new ListBuffer[Seq[Double]]; //每一元素都是一个属性的所有取值
        for (i <- Range(0, X.cols)) {
            val Xcol: vector[Double] = X(::, i)
            val Xset = new mutable.HashSet[Double]()
            Xcol.foreach(xi => Xset.add(xi))
            XVals.append(Xset.toSeq)
        }
        val samples = Range(0, X.rows)
        val root = new Node(samples, 0)

        this.Attributes = XVals.toList
        buildNode(root)

        prune(Option(root))

        this.dt = root

        val Ytest = test(X).toDenseVector
        (this.dt, Ytest, loss(Y.toDenseVector, Ytest))
    }

    def buildNode(node: Node): Unit = {
        val Xvals = this.Attributes

        val yClasses = classifyY(node.samples)
        var yClass: Double = 0.0
        var max = 0.0
        for (i <- yClasses.toSeq) {
            if (i._2 > max) max = i._2
            yClass = i._1
        }
        node.cls = Option(yClass)
        node.gini = Option(calGini(node.samples))

        if (node.depth < maxDepth && node.samples.size > samplesThreshold && calGini(node.samples) > giniThreshold) {

            var GiniWithA = 1.0
            var left: Seq[Int] = null
            var right: Seq[Int] = null
            var iColRecord = -1
            var valueRecord: Double = -1
            for (iCol <- Xvals.indices) {
                for (j <- Xvals(iCol).indices) {
                    val tmp = calGiniWithA(iCol, Xvals(iCol)(j), node)
                    if (tmp._1 < GiniWithA) {
                        GiniWithA = tmp._1
                        left = tmp._2
                        right = tmp._3
                        iColRecord = iCol
                        valueRecord = Xvals(iCol)(j)
                    }
                }
            }
            node.attribute = Option(iColRecord)
            node.value = Option(valueRecord)

            if (left.size > 0) {
                node.leftNode = Option(new Node(left, node.depth + 1))
                buildNode(node.leftNode.get)
            }
            if (right.size > 0) {
                node.rightNode = Option(new Node(right, node.depth + 1))
                buildNode(node.rightNode.get)
            }
        }
    }

    def calGiniWithA(iCol: Int, value: Double, node: Node) = {
        val trueSet = new mutable.HashSet[Int]() //record index of sample
        val falseSet = new mutable.HashSet[Int]()
        for (i <- node.samples.indices) {
            val index = node.samples(i)
            //            if (X(index, iCol) == value) {
            if (X(index, iCol) <= value) {
                trueSet.add(index)
            } else {
                falseSet.add(index)
            }
        }
        val gini = ((trueSet.size.toDouble / node.samples.size) * calGini(trueSet.toSeq)
            + (falseSet.size.toDouble / node.samples.size) * calGini(falseSet.toSeq))

        (gini, trueSet.toSeq, falseSet.toSeq)
    }

    def calGini(indexes: Seq[Int]) = {
        val map = classifyY(indexes)
        val pk2 = map.values.map(xi => math.pow(xi / indexes.size.toDouble, 2)).sum
        1 - pk2
    }

    def classifyY(indexes: Seq[Int]) = {
        val map = new mutable.HashMap[Double, Double]()
        for (i <- indexes.indices) {
            val y = Y(indexes(i))
            if (!map.contains(y)) map.put(y, 0)

            map.put(y, map.get(y).get + 1)
        }
        map
    }

    def prune(option:Option[Node]):Unit = {
        if(option.isEmpty) return
        val node = option.get
        if (!node.isLeaf()) {
            prune(node.leftNode)
            prune(node.rightNode)

            if(node.leftNode.isDefined && node.rightNode.isDefined &&
                node.leftNode.get.isLeaf() && node.rightNode.get.isLeaf()) {
                if (node.gini.get -
                    (node.leftNode.get.gini.get) * (node.leftNode.get.samples.size)/node.samples.size.toDouble
                    - (node.rightNode.get.gini.get) * (node.rightNode.get.samples.size)/node.samples.size.toDouble
                < this.gama){
                    node.leftNode = None
                    node.rightNode = None
                }
            }
        }
    }

    def test(X: matrix[Double]) = {
        val res = new ListBuffer[Double]
        for (i <- Range(0, X.rows)) {
            res append decideCls(X(i, ::).t)
        }
        matrix.tabulate(res.size, 1) { case (i, j) => res(i) }
    }

    def decideCls(row: vector[Double]): Double = {
        decideNode(row, this.dt)
    }

    def decideNode(row: vector[Double], node: Node): Double = {
        if (node.leftNode.isEmpty && node.rightNode.isEmpty) return node.cls.get

        //        if (row(node.attribute) == node.value)
        if (row(node.attribute.get) <= node.value.get) {
            if (node.leftNode != null) decideNode(row, node.leftNode.get)
            else node.cls.get
        } else {
            if (node.rightNode != null) decideNode(row, node.rightNode.get)
            else node.cls.get
        }
    }

}

class Node(val samples: Seq[Int], val depth: Int) {
    //samples records indexes of samples
    var leftNode: Option[Node] = None
    var rightNode: Option[Node] = None
    var cls: Option[Double] = None
    var attribute: Option[Int] = None
    var value: Option[Double] = None
    var gini: Option[Double] = None
    var giniWithA: Option[Double] = None

    def isLeaf():Boolean = {
        leftNode.isEmpty && rightNode.isEmpty
    }
}

object DecisionTree {

    def main(args: Array[String]) {
        val path = "iris.csv"
        val rawData = breeze.linalg.csvread(new File("./data/" + path), skipLines = 1);
        val Y = rawData(::, rawData.cols - 1).toDenseMatrix.t
        val X = matrix.tabulate(rawData.rows, rawData.cols - 1) { case (i, j) => rawData(i, j) }

        val samplesTest = new ListBuffer[Int]
        val slice = 5
        for (i <- Range(0, X.rows / slice)) {
            samplesTest.append(i * slice + (math.random * slice).toInt)
        }

        val listTest = new ListBuffer[vector[Double]]
        for (i: Int <- Range(0, samplesTest.size)) {
            listTest.append(X(samplesTest(i), ::).t)
        }
        val Xtest = matrix.tabulate(listTest.size, X.cols) { case (i, j) => listTest(i)(j) }
        val Ytest = matrix.tabulate(listTest.size, 1) { case (i, j) => Y(samplesTest(i), 0) }

        val samplesTrain = Range(0, X.rows).toSet -- samplesTest.toSet toList
        val listTrain = new ListBuffer[vector[Double]]
        for (i: Int <- Range(0, samplesTrain.size)) {
            listTrain.append(X(samplesTrain(i), ::).t)
        }
        val Xtrain = matrix.tabulate(listTrain.size, X.cols) { case (i, j) => listTrain(i)(j) }
        val Ytrain = matrix.tabulate(listTrain.size, 1) { case (i, j) => Y(samplesTrain(i), 0) }

        val dt = new DecisionTree
        dt.train(Xtrain, Ytrain.toDenseVector)

        val YTrainRes = dt.test(Xtrain)
        println(Common.loss(Ytrain.toDenseVector, YTrainRes.toDenseVector))
//        val comp2 = matrix.tabulate(YTrainRes.rows, 1) { case (i, j) => if (Ytrain(i, 0) == YTrainRes(i, 0)) 1; else 0 }
//        println(sum(comp2).toDouble / Ytrain.rows)

        val YTestRes = dt.test(Xtest)
        println(Common.loss(YTestRes.toDenseVector, Ytest.toDenseVector))
//        val comp = matrix.tabulate(Ytest.rows, 1) { case (i, j) => if (Ytest(i, 0) == YTestRes(i, 0)) 1; else 0 }
//        println(sum(comp).toDouble / Ytest.rows)
    }
}