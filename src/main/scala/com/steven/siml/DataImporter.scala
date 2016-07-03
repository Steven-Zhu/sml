package com.steven.siml

import breeze.linalg.{DenseMatrix => matrix}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer


object DataImporter {

    def readDense(file: Iterator[String]): (matrix[Double], matrix[Double]) = {
        val X: ListBuffer[List[Double]] = ListBuffer()
        val Y: ListBuffer[Int] = ListBuffer()

        var maxLen = 0
        for (line <- file) {
            val items = line.split(" ")
            Y.append(items(0).toInt)
            var x = Array[Double]()
            for (item <- items.takeRight(items.length - 1)) {
                val kv = item.split(":")

                if (kv(0).toInt > maxLen) maxLen = kv(0).toInt
                val tmp = x
                x = new Array[Double](maxLen)
                tmp.copyToArray(x)
                x(kv(0).toInt - 1) = kv(1).toDouble
            }
            X.append(x.toList)
        }
        //返回的X看做是行向量构成的， Y则是列向量
        (matrix.tabulate(X.length, X(0).length) { case (i, j) => X(i)(j) },
            matrix.tabulate(Y.length, 1) { case (i, j) => Y(i) })
    }

    def readsparse(file: Iterator[String]): (Seq[Seq[(String, Int)]], Seq[String]) = {
        val res = file.map(line => {
            val items = line.split(" ")
            val y = items(0)
            val xItems = items.takeRight(items.length - 1)
            val x = xItems.map(xi => {
                val h = xi.split(":"); (h(0), h(1).toInt)
            }).toSeq
            (x, y)
        }).filter(_._1.length != 0)
        val X = new mutable.ListBuffer[Seq[(String, Int)]]
        val Y = new mutable.ListBuffer[String]
        res.foreach(k => {
            X.append(k._1); Y.append(k._2)
        })
        (X, Y)
    }
}
