package de.heikospindler.spark.classifier.mnist;


import de.heikospindler.spark.Const;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;

public class JavaMnistRF {

    public static void main(String[] args) throws Exception {

        // Create a spark session with a name and two local nodes
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMNISTDT")
                .master("local[2]")
                .getOrCreate();

        // We have 784 pixel for every bitmap
        int PIXEL_COUNT = 784;

        // Create an array with the names of the pixel columns p0..p783
        String[] inputFeatures = new String[PIXEL_COUNT];
        for ( int index = 0; index < PIXEL_COUNT; index ++) {
            inputFeatures[index] = "p"+index;
        }

        Dataset<Row> trainFull = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", true)
                .format("com.databricks.spark.csv")
                .load(Const.BASE_DIR_DATASETS+"mnist_train2.csv");

        Dataset<Row> testFull = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", true)
                .format("com.databricks.spark.csv")
                .load(Const.BASE_DIR_DATASETS+"mnist_test2.csv");

        StructType schema2 = testFull.schema();

        JavaRDD<Row> trainJavaRS = trainFull.toJavaRDD().filter(e ->  Math.random() > 0.90 );
        JavaRDD<Row> testJavaRS = testFull.toJavaRDD().filter(e ->  Math.random() > 0.90 );

        Dataset<Row> train2 = spark.createDataFrame(trainJavaRS, schema2);
        Dataset<Row> test2 = spark.createDataFrame(testJavaRS, schema2);

        SQLTransformer sqlTranslabel = new SQLTransformer().setStatement(
                "SELECT * , (1.0 * label) AS label2 FROM __THIS__ ");
        Dataset<Row> train = sqlTranslabel.transform(train2);
        Dataset<Row> test = sqlTranslabel.transform(test2);


        System.out.println( "Using training entries: "+train.count());
        System.out.println( "Using test entries: "+test.count());

//        train.printSchema();
//        train.show(5);

        // Create the workflow steps
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols( inputFeatures )
                .setOutputCol("features");

        StringIndexerModel stringIndexer = new StringIndexer()
                .setInputCol("label")
                .setHandleInvalid("skip")
                .setOutputCol("indexedLabel")
                .fit(train)
                ;

        // Train a DecisionTree classifier.
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setMaxDepth(25)
                .setLabelCol("indexedLabel")
                .setFeaturesCol(assembler.getOutputCol());

        RandomForestClassifier rf = new RandomForestClassifier()
                .setMaxDepth(25).setNumTrees(30)
                .setLabelCol("indexedLabel")
                .setFeaturesCol(assembler.getOutputCol());

        //

        IndexToString indexToString = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(stringIndexer.labels());

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {
                        assembler
                        , stringIndexer
//                        , dt
                        , rf
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

        SQLTransformer sqlTrans3 = new SQLTransformer().setStatement(
                "SELECT (100.0*count(*)/"+test.count()+") AS Correct FROM __THIS__ where label2 = prediction");
        Dataset<Row> df4 = sqlTrans3.transform(result);
        df4.show();


//        String[] labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        List<Object> labels2 = new ArrayList<Object>();
        for (String s : "0,1,2,3,4,5,6,7,8,9".split(","))
        {
            labels2.add( s );
        }

        String s = result.select(new Column("label2"), new Column("prediction"))
                        .orderBy("label2")
                        .groupBy("label2")
                        .pivot("prediction",labels2)
                        .count()
                        .showString(10, false);

        // Just for better reading
        System.out.println( s.replace("null", "    "));

        System.out.println( "Modeltraining and Test took (ms) : "+(time2-time1));
    }
}
