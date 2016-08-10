/*
 * Copyright 2015 Sanford Ryza, Uri Laserson, Sean Owen and Joshua Wills
 *
 * See LICENSE file for further information.
 */

package com.colobu.douban.recommender

import scala.collection.Map

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

case class MovieRating(userID: String, movieID: Int, rating: Double) extends scala.Serializable

object DoubanRecommender {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("DoubanRecommender"))
    //val base = "/opt/douban/"
    val base = if (args.length > 0) args(0) else "/opt/douban/"

    //获取RDD
    val rawUserMoviesData = sc.textFile(base + "user_movies.csv")
    val rawHotMoviesData = sc.textFile(base + "hot_movies.csv")

    //准备数据
    preparation(rawUserMoviesData, rawHotMoviesData)
    println("准备完数据")

    //model(sc, rawUserMoviesData, rawHotMoviesData)

    //evaluate(sc,rawUserMoviesData, rawHotMoviesData)

    recommend(sc, rawUserMoviesData, rawHotMoviesData,base)
  }

  //得到电影名字的RDD
  def buildMovies(rawHotMoviesData: RDD[String]): Map[Int, String] =
    rawHotMoviesData.flatMap { line =>
      val tokens = line.split(',')
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(2)))
      }
    }.collectAsMap()


  //分析清理数据
  def preparation( rawUserMoviesData: RDD[String],
                   rawHotMoviesData: RDD[String]) = {
    val userIDStats = rawUserMoviesData.map(_.split(',')(0).trim).distinct().zipWithUniqueId().map(_._2.toDouble).stats()
    val itemIDStats = rawUserMoviesData.map(_.split(',')(1).trim.toDouble).distinct().stats()
    println(userIDStats)
    println(itemIDStats)

    val moviesAndName = buildMovies(rawHotMoviesData)

    val (movieID, movieName) = moviesAndName.head
    println(movieID + " -> " + movieName)
  }

  //
  def buildRatings(rawUserMoviesData: RDD[String]): RDD[MovieRating] = {
    rawUserMoviesData.map { line =>
      val Array(userID, moviesID, countStr) = line.split(',').map(_.trim)
      var count = countStr.toInt
      count = if (count == -1) 3 else count
      MovieRating(userID, moviesID.toInt, count)
    }
  }

  //http://stackoverflow.com/questions/27772769/spark-how-to-use-mllib-recommendation-if-the-user-ids-are-string-instead-of-co
  def model(sc: SparkContext,
            rawUserMoviesData: RDD[String],
            rawHotMoviesData: RDD[String]): Unit = {

    val moviesAndName = buildMovies(rawHotMoviesData)
    val bMoviesAndName = sc.broadcast(moviesAndName)

    val data = buildRatings(rawUserMoviesData)

    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()
    val reverseUserIDMapping: RDD[(Long, String)] =
      userIdToInt map { case (l, r) => (r, l) }

    val userIDMap: Map[String, Int] =
      userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }

    val bUserIDMap = sc.broadcast(userIDMap)

    val ratings: RDD[Rating] = data.map { r =>
      Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
    }.cache()
    //使用协同过滤算法建模
    //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
    val model = ALS.train(ratings, 50, 10, 0.0001)
    ratings.unpersist()
    println("打印第一个userFeature")
    println(model.userFeatures.mapValues(_.mkString(", ")).first())

    for (userID <- Array(100,1001,10001,100001,110000)) {
      checkRecommenderResult(userID, rawUserMoviesData, bMoviesAndName, reverseUserIDMapping, model)
    }

    unpersist(model)
  }

  //
  //查看给某个用户的推荐
  def checkRecommenderResult(userID: Int, rawUserMoviesData: RDD[String], bMoviesAndName: Broadcast[Map[Int, String]], reverseUserIDMapping: RDD[(Long, String)], model: MatrixFactorizationModel): Unit = {

    val userName = reverseUserIDMapping.lookup(userID).head

    val recommendations = model.recommendProducts(userID, 5)
    //给此用户的推荐的电影ID集合
    val recommendedMovieIDs = recommendations.map(_.product).toSet

    //得到用户点播的电影ID集合
    val rawMoviesForUser = rawUserMoviesData.map(_.split(',')).
      filter { case Array(user, _, _) => user.trim == userName }
    val existingUserMovieIDs = rawMoviesForUser.map { case Array(_, movieID, _) => movieID.toInt }.
      collect().toSet


    println("用户" + userName + "点播过的电影名")
    //点播的电影名
    bMoviesAndName.value.filter { case (id, name) => existingUserMovieIDs.contains(id) }.values.foreach(println)

    println("推荐给用户" + userName + "的电影名")
    //推荐的电影名
    bMoviesAndName.value.filter { case (id, name) => recommendedMovieIDs.contains(id) }.values.foreach(println)
  }


  def evaluate( sc: SparkContext,
                rawUserMoviesData: RDD[String],
                rawHotMoviesData: RDD[String]): Unit = {
    val moviesAndName = buildMovies(rawHotMoviesData)
    val data = buildRatings(rawUserMoviesData)

    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()


    val userIDMap: Map[String, Int] =
      userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }

    val bUserIDMap = sc.broadcast(userIDMap)

    val ratings: RDD[Rating] = data.map { r =>
      Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
    }.cache()

    val numIterations = 10

    for (rank   <- Array(10,  50);
         lambda <- Array(1.0, 0.01,0.0001)) {
      val model = ALS.train(ratings, rank, numIterations, lambda)

      // Evaluate the model on rating data
      val usersMovies = ratings.map { case Rating(user, movie, rate) =>
        (user, movie)
      }
      val predictions =
        model.predict(usersMovies).map { case Rating(user, movie, rate) =>
          ((user, movie), rate)
        }
      val ratesAndPreds = ratings.map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }.join(predictions)

      val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      println(s"(rank:$rank, lambda: $lambda, Explicit ) Mean Squared Error = " + MSE)
    }

    for (rank   <- Array(10,  50);
         lambda <- Array(1.0, 0.01,0.0001);
         alpha  <- Array(1.0, 40.0)) {
      val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, alpha)

      // Evaluate the model on rating data
      val usersMovies = ratings.map { case Rating(user, movie, rate) =>
        (user, movie)
      }
      val predictions =
        model.predict(usersMovies).map { case Rating(user, movie, rate) =>
          ((user, movie), rate)
        }
      val ratesAndPreds = ratings.map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }.join(predictions)

      val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      println(s"(rank:$rank, lambda: $lambda,alpha:$alpha ,implicit  ) Mean Squared Error = " + MSE)
    }
  }

  def recommend(sc: SparkContext,
                rawUserMoviesData: RDD[String],
                rawHotMoviesData: RDD[String],
                base:String): Unit = {
    val moviesAndName = buildMovies(rawHotMoviesData)
    val bMoviesAndName = sc.broadcast(moviesAndName)

    val data = buildRatings(rawUserMoviesData)

    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()
    val reverseUserIDMapping: RDD[(Long, String)] =
      userIdToInt map { case (l, r) => (r, l) }

    val userIDMap: Map[String, Int] =
      userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }

    val bUserIDMap = sc.broadcast(userIDMap)
    val bReverseUserIDMap = sc.broadcast(reverseUserIDMapping.collectAsMap())

    val ratings: RDD[Rating] = data.map { r =>
      Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
    }.cache()
    //使用协同过滤算法建模
    //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
    val model = ALS.train(ratings, 50, 10, 0.0001)
    ratings.unpersist()

    model.save(sc, base+"model")
    //val sameModel = MatrixFactorizationModel.load(sc, base + "model")

    val allRecommendations = model.recommendProductsForUsers(5) map {
      case (userID, recommendations) => {
        var recommendationStr = ""
        for (r <- recommendations) {
          recommendationStr += r.product + ":" + bMoviesAndName.value.getOrElse(r.product, "") + ","
        }
        if (recommendationStr.endsWith(","))
          recommendationStr = recommendationStr.substring(0,recommendationStr.length-1)

        (bReverseUserIDMap.value.get(userID).get,recommendationStr)
      }
    }

    //allRecommendations.saveAsTextFile(base + "result.csv")
    allRecommendations.coalesce(1).sortByKey().saveAsTextFile(base + "result.csv")

    unpersist(model)
  }

  def unpersist(model: MatrixFactorizationModel): Unit = {
    // At the moment, it's necessary to manually unpersist the RDDs inside the model
    // when done with it in order to make sure they are promptly uncached
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()
  }

}