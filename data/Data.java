package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

// can be used both for training and validating
// (convenient if you want to validate on the trainset...)
public class Data {

    private List<FeatureVector> featureVectors;
    private String[] featureIdsToNames;
    private boolean fitted = false; // will be true if either it has been used for training or for validation

    public List<FeatureVector> getFeatureVectors() { return featureVectors; }
    public int getNumFeatureVectors() { return featureVectors.size(); }
    public int getNumFeatures() { return featureIdsToNames.length; }
    
    
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < featureIdsToNames.length; i++) {
            builder.append(featureIdsToNames[i]);
            if (i < featureIdsToNames.length - 1)
                builder.append(",");
            else
                builder.append(",");
        }
        builder.append("Label,Prob");
        builder.append("\n");
        for (FeatureVector vector : featureVectors) {
            builder.append(vector.toString());
            builder.append("\n");
        }
        return builder.toString();
    }

    // File must contain Label column as well as feature columns
    // All features must be numeric. Nulls are NOT allowed.

    /* EXAMPLE FORMAT:
    * Label,FeatureA,FeatureB,FeatureC
    * 1,52.4,0.98,-1.77
    * 0,999.9,0.42,3.8
    */
    @SuppressWarnings("resource")
	public Data(String filepath, String labelName) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filepath));

        String nextLine = reader.readLine();

        String[] fullHeaderWords = nextLine.split(",");
        List<String> fullHeaderWordsList = Arrays.asList(fullHeaderWords);
        int labelIndex = fullHeaderWordsList.indexOf(labelName);

        if (labelIndex == -1)
            throw new ArrayIndexOutOfBoundsException("Label field does not exist.");


        featureIdsToNames = new String[fullHeaderWords.length - 1];

        int targetCol = 0;
        for (int col = 0; col < fullHeaderWords.length; col++) {
            if (col != labelIndex) {
                featureIdsToNames[targetCol] = fullHeaderWords[col];
                targetCol++;
            }
        }

        nextLine = reader.readLine();

        featureVectors = new ArrayList<>();

        while(nextLine != null) {
            String[] featureValueWords = nextLine.split(",");

            double[] featureValues = new double[featureValueWords.length - 1]; // -1 to exclude the label column

            // parsing label with error-handling
            int labelAsInt = Integer.parseInt(featureValueWords[labelIndex]);
            boolean label;
            if (labelAsInt == 1)
                label = true;
            else if (labelAsInt == 0)
                label = false;
            else
                throw new NumberFormatException("Labels must be 1 or 0.");

            targetCol = 0;
            for (int col = 0; col < fullHeaderWords.length; col++) {
                if (col != labelIndex) {
                    featureValues[targetCol] = Double.parseDouble(featureValueWords[col]);
                    targetCol++;
                }
            }
            FeatureVector vector = new FeatureVector(label, featureValues);
            featureVectors.add(vector);
            nextLine = reader.readLine();
        }

        reader.close();
    }

    

    public void save(String filepath) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filepath));
        writer.write(this.toString());
        writer.close();
    }
    
    
    // prints precision-recall evaluations at each of the precisions asked for
    public void evaluate(double[] inputPrecisions) {
    	
        Collections.sort(featureVectors);

        double totalCountPositive = (double) featureVectors.stream().filter(datapoint -> datapoint.getLabel()).count();

        List<Double> thresholds = new ArrayList<>();
        List<Double> precisions = new ArrayList<>();
        List<Double> recalls = new ArrayList<>();

        double currentThreshold = 1.0;
        double currentCountAboveThreshold = 0.0;
        double currentCountPositiveAboveThreshold = 0.0;

        thresholds.add(1.0);
        precisions.add(1.0);
        recalls.add(0.0);

        for (FeatureVector vector : featureVectors) {
            currentThreshold = vector.getProb();
            currentCountAboveThreshold += 1.0;
            if (vector.getLabel()) {
                currentCountPositiveAboveThreshold += 1.0;
            }

            thresholds.add(currentThreshold);
            precisions.add(currentCountPositiveAboveThreshold / currentCountAboveThreshold);
            recalls.add(currentCountPositiveAboveThreshold / totalCountPositive);

        }
        
        for (double precisionTarget : inputPrecisions) {
        	
            int bestViableIndex = 0;

            for (int index = 0; index < precisions.size(); index++) {
                if (precisions.get(index) >= precisionTarget) {
                    bestViableIndex = index;
                }
            }

            double thresholdChosen = thresholds.get(bestViableIndex);
            double recallAchieved = recalls.get(bestViableIndex);
            double actualPrecisionAchieved = precisions.get(bestViableIndex); // might be slightly off from the desired precision

            StringBuilder builder = new StringBuilder();
            builder.append("Precision: ");
            builder.append(String.format("%.3f", actualPrecisionAchieved));
            builder.append(", Recall: ");
            builder.append(String.format("%.3f", recallAchieved));
            builder.append(", Threshold: ");
            builder.append(String.format("%.3f", thresholdChosen));
            System.out.println(builder);
            
        }
    }
    
    
    public void markAsFitted() {
    	fitted = true;
    }
    
    // erase from previous train or predict
    public void clearLogits() {
    	if (fitted) {
    		featureVectors.forEach(vector -> {vector.clearLogit();});
    	}
    	fitted = false;
    }

}
