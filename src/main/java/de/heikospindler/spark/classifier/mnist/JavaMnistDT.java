package de.heikospindler.spark.classifier.mnist;


import de.heikospindler.spark.Const;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;

import java.util.Arrays;
import java.util.List;

public class JavaMnistDT {

    // We have 784 pixel for every bitmap
    static final int PIXEL_COUNT = 784;

    public static void main(String[] args) throws Exception {

        // Create a spark session with a name and two local nodes
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMNISTDT")
                .master("local[3]")
                .getOrCreate();

        // Prepare training and test data.
        DataFrameReader reader = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", true)
                .format("com.databricks.spark.csv");

        Dataset<Row> test = reader
                .load(Const.BASE_DIR_DATASETS+"mnist_test2.csv")
                .filter(e ->  Math.random() > 0.00 );

        Dataset<Row> train = reader
                .load(Const.BASE_DIR_DATASETS+"mnist_train2.csv")
                .filter(e ->  Math.random() > 0.00 );

        System.out.println( "Using training entries: "+train.count());
        System.out.println( "Using test entries: "+test.count());

//        train.printSchema();
//        train.show(5);

        // Create the workflow steps

        // Create array with the names of the pixel columns p0..p783 as input to the feature vector
        String[] inputFeatures = new String[PIXEL_COUNT];
        for ( int index = 0; index < PIXEL_COUNT; index ++) {
            inputFeatures[index] = "p"+index;
        }
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols( inputFeatures )
                .setOutputCol("features");

        StringIndexerModel stringIndexer = new StringIndexer()
                .setInputCol("label")
                .setHandleInvalid("skip")
                .setOutputCol("indexedLabel")
                .fit(train);

        // Train a DecisionTree classifier.
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setMaxDepth(30).setSeed(12345L)
                .setLabelCol(stringIndexer.getOutputCol())
                .setFeaturesCol(assembler.getOutputCol());

        IndexToString indexToString = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(stringIndexer.labels());

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {
                        assembler
                        , stringIndexer
                        , dt
                        , indexToString
                });

        System.out.println( "Training ----------");
        long time1 = System.currentTimeMillis();

        // Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(train);

        // Use the model to evaluate the test set.
        Dataset<Row> result = model.transform(test);

        long time2 = System.currentTimeMillis();
        result.show();

        System.out.println( "Results ----------");
        System.out.println( "Correct:"+ (100.0 * result.filter("label = predictedLabel").count() / result.count()) );

        // Create a convolution matrix
        List<Object> labelNames = Arrays.asList("0,1,2,3,4,5,6,7,8,9".split(","));

        String matrix = result.select(new Column(stringIndexer.getInputCol()), new Column(indexToString.getOutputCol()))
                        .orderBy(stringIndexer.getInputCol())
                        .groupBy(stringIndexer.getInputCol())
                        .pivot(indexToString.getOutputCol(),labelNames)
                        .count()
                        .showString(10, 0)
                        .replace("null", "    ");

        System.out.println( matrix );

        System.out.println( "Model training and Test took (sec.) : "+(time2-time1)/1000);

    }
}
