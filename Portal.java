import model.Config;
import model.Data;
import model.GBTModel;

// Use this as an entry-point for scoring real data
public class Portal {

    public static void main(String[] args) {

        try {
            Data traindata = Data.load("DataSets/trainset.csv", "Label");
            Data testdata = Data.load("DataSets/testset.csv", "Label");

            System.out.println("Data loaded");

            Config config = Config.builder()
                    .setNumTrees(100)
                    .setMaxTreeDepth(6)
                    .setMinSamplesLeaf(25)
                    .setNumFeaturesSplit(10)
                    .setLearningRate(0.1)
                    .setL2reg(0.1)
                    .setNumThreads(3)
                    .build();

            System.out.println(config.toString());

            long trainStart = System.currentTimeMillis();
            GBTModel model = GBTModel.train(config, traindata);
            long trainEnd = System.currentTimeMillis();
            System.out.println("Train complete. Time = " + Long.toString(trainEnd - trainStart) + " millis");
            
            model.printFeatureImportances();

            long predictStart = System.currentTimeMillis();
            model.predict(testdata);
            long predictEnd = System.currentTimeMillis();
            System.out.println("Predict complete. Time = " + Long.toString(predictEnd - predictStart) + " millis");
            
            testdata.evaluate( new double[] { 0.7, 0.8, 0.9 } );
            testdata.save("DataSets/scores.csv");
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
