package de.heikospindler.spark.cluster;

import de.heikospindler.spark.Const;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class KmeansSample {

  public static void main(String[] args) {

    SparkConf conf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local");
   
    JavaSparkContext jsc = new JavaSparkContext(conf);

    // Load and parse data
    JavaRDD<String> data = jsc.textFile(Const.BASE_DIR_DATASETS+"clusterdata.csv");
    JavaRDD<Vector> parsedData = data.map(
      new Function<String, Vector>() {
        public Vector call(String s) {
          String[] sarray = s.split(",");
          double[] values = new double[sarray.length];
          for (int i = 0; i < sarray.length; i++) {
            values[i] = Double.parseDouble(sarray[i]);
          }
          return Vectors.dense(values);
        }
      }
    );
    parsedData.cache();

    System.out.println("Input vectors:");
    for (Vector vec: parsedData.take(5)) {
      System.out.println(" " + vec);
    }

    // Cluster the data into two classes using KMeans

    // === Try other values for the number of clusters
    int numClusters = 3;
    int numIterations = 2000;
    KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

    System.out.println("Cluster centers:");
    for (Vector center: clusters.clusterCenters()) {
      System.out.println(" " + center);
    }
    double cost = clusters.computeCost(parsedData.rdd());
    System.out.println("Cost: " + cost);


    // === Predict a new vector with the generated model
    double[]  testData =  { 30.0, 10.5 };
    Vector testVector = new DenseVector(testData);
    int clusterIndex = clusters.predict( testVector );
    System.out.println("Vector "+testVector.toString()+" belongs to cluster index:"+ clusterIndex);

    jsc.stop();
  }
}