package de.heikospindler.spark.classifier.mnist;

import de.heikospindler.spark.Const;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class JavaMnistMLPC_Binarizer {

    public static void main(String[] args) throws Exception {

        // Create a spark session with a name and two local nodes
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaMNISTMLPC_Binarizer")
                .master("local[3]")
                .getOrCreate();

        // Define the neuronal net work topology
        int[] layers = { 784, 800, 100,  10 };

        // Prepare training and test data.
        DataFrameReader reader = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", true)
                .format("com.databricks.spark.csv");

        Dataset<Row> test = reader
                .load(Const.BASE_DIR_DATASETS+"mnist_test2.csv")
                .filter((FilterFunction)(e ->  Math.random() > 0.00 ));

        Dataset<Row> train = reader
                .load(Const.BASE_DIR_DATASETS+"mnist_train2.csv")
                .filter((FilterFunction)(e ->  Math.random() > 0.00 ));

        System.out.println( "Using training entries: "+train.count());
        System.out.println( "Using test entries: "+test.count());

//        train.printSchema();
//        train.show(5);

        // Create array with the names of the pixel columns p0..p783 as input to the feature vector
        StructType schema = train.schema();
        String[] inputFeatures = Arrays.copyOfRange(schema.fieldNames(), 1, schema.fieldNames().length);

        // Create the ml pipeline elements
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols( inputFeatures )
                .setOutputCol("features");

        Binarizer binarizer = new Binarizer()
                .setInputCol("features")
                .setOutputCol("bin_feature")
                .setThreshold(127.5);

        StringIndexerModel stringIndexer = new StringIndexer()
                .setInputCol("label")
                .setHandleInvalid("skip")
                .setOutputCol("indexedLabel")
                .fit(train);

        MultilayerPerceptronClassifier mlpc = new MultilayerPerceptronClassifier()
                .setLabelCol(stringIndexer.getOutputCol())
                .setFeaturesCol(binarizer.getOutputCol())
                .setLayers(layers)
                .setSeed(12345L)
                .setBlockSize(128) //default 128
                .setMaxIter(1000) //default 100
//                .setMaxIter(10000) //default 100
                .setTol(1e-7) //default 1e-6
//                .setSolver("GD") // l-bfgs or gd
                .setStepSize(0.02); // Default 0.03

        IndexToString indexToString = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(stringIndexer.labels());

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {assembler
                        , stringIndexer
                        , binarizer
                        , mlpc
                        , indexToString
                });

        System.out.println( "Training ----------");
        long time1 = System.currentTimeMillis();

        // Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(train);

        Dataset<Row> result = model.transform(test);

        long time2 = System.currentTimeMillis();
        result.show();
        System.out.println( "Results ----------");
        SQLTransformer sqlTrans3 = new SQLTransformer().setStatement(
                "SELECT 100.0*count(*)/"+test.count()+" as Correct FROM __THIS__ where label = predictedLabel");
        Dataset<Row> df4 = sqlTrans3.transform(result);
        df4.show();

        // Create a convolution matrix
        List<Object> labelNames = Arrays.asList("0,1,2,3,4,5,6,7,8,9".split(","));


        String matrix = result.select(new Column(stringIndexer.getInputCol()), new Column(indexToString.getOutputCol()))
                .orderBy(stringIndexer.getInputCol())
                .groupBy(stringIndexer.getInputCol())
                .pivot(indexToString.getOutputCol(),labelNames)
                .count()
                .showString(10, 0, false)
                .replace("null", "    ");

        // Just for better reading
        System.out.println( matrix );

        System.out.println( "Model training and Test took (min.) : "+(time2-time1)/(1000*60));
    }

}
