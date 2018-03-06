package node;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import data.FeatureVector;
import model.Config;

// calculates and holds the best split point for one specific feature, and for one specific leaf sample taken from one specific bag
public class Split implements Callable<BranchNode> {
	
	private Config config;
	private int featureId;
	private List<FeatureVector> datapoints; // this is a reference to a list shared by many threads
	private int depth;
	
	public Split(Config config, int featureId, List<FeatureVector> datapoints, int depth) {
		this.config = config;
		this.featureId = featureId;
		this.datapoints = datapoints;
		this.depth = depth;
	}

    // given the config, the features and the datapoints in the leafnode, returns the best split point
	@Override
    public BranchNode call() {
		
		datapoints = new ArrayList<>(datapoints);  // make shallow copy, because (i) we are accessing datapoints from multiple threads
		// ... and (ii) because datapoints might be a view of a sublist
		// doing this inside the thread
    	
        int totalSamples = datapoints.size();
         
        // sort the samples in the leaf by the value of the chosen feature
        datapoints.sort((vector1, vector2) -> Double.compare(vector1.getFeatureValue(featureId), vector2.getFeatureValue(featureId)));
        
        // initially, everything is sent to the right.
        double sumLeftFirstDerivs = 0.0;
        double sumLeftSecondDerivs = 0.0;
        double sumRightFirstDerivs = 0.0;
        double sumRightSecondDerivs = 0.0;
        for (FeatureVector vector : datapoints) {
        	sumRightFirstDerivs += vector.getFirstDeriv(); // do not use .stream() here - too slow!
        	sumRightSecondDerivs += vector.getSecondDeriv();
        }

        int currentSplitPosition = 0; // NB currentSplitPosition will always be equal to the number of samples sent to the LEFT.

        double previousValue = datapoints.get(currentSplitPosition).getFeatureValue(featureId);
        
        double entropyDecreaseWithoutSplit = - 0.5 * sumRightFirstDerivs * sumRightFirstDerivs / sumRightSecondDerivs;
        double bestEntropyDecrease = entropyDecreaseWithoutSplit;
        	// this is the benchmark, i.e. what would be achieved without splitting
        Double bestSplitThreshold = null;
        Integer bestSplitPosition = null;

        // now transfer the elements across from the right to the left, one by one, starting with the lowest ranked one
        while(true) {
            // e.g. if the ordered values are 4.1, 4.1, 5.2, 5.2, 5.2, 7.3, 10.5, then previousValue will initially be 4.1.
            // ... and after the mini while loop, it will look like: 4.1, 4.1 | 5.2, 5.2, 5.2, 7.3, 10.5.
            // The next time round, it will look like 4.1, 4.1 | 5.2, 5.2, 5.2, 7.3, 10.5, with initial value now bumped up to 5.2
            // ... and after the mini while loop, it will look like 4.1, 4.1, 5.2, 5.2, 5.2 | 7.3, 10.5
            while(currentSplitPosition < totalSamples && datapoints.get(currentSplitPosition).getFeatureValue(featureId) == previousValue) {
            	
                double currentFirstDeriv = datapoints.get(currentSplitPosition).getFirstDeriv();
                double currentSecondDeriv = datapoints.get(currentSplitPosition).getSecondDeriv();
                
                sumLeftFirstDerivs += currentFirstDeriv;
                sumRightFirstDerivs -= currentFirstDeriv;
                sumLeftSecondDerivs += currentSecondDeriv;
                sumRightSecondDerivs -= currentSecondDeriv;

                currentSplitPosition ++;
            }

            // If the number of samples remaining in the right leaf is less than minSamplesLeaf, then there is no point in continuing
            if (currentSplitPosition > totalSamples - config.getMinSamplesLeaf()) {
                break;
            }

            // Only consider the possibility of splitting if the number of samples in the left leaf is at least as large as minSamplesLeaf
            if (currentSplitPosition >= config.getMinSamplesLeaf()) {
            	
                // Calculate metric gain if splitting here
            	double leftEntropyDecrease = - 0.5 * sumLeftFirstDerivs * sumLeftFirstDerivs / sumLeftSecondDerivs;
            	double rightEntropyDecrease = - 0.5 * sumRightFirstDerivs * sumRightFirstDerivs / sumRightSecondDerivs;
                double entropyDecrease = leftEntropyDecrease + rightEntropyDecrease;
                
                boolean notNaN = !Double.isNaN(entropyDecrease); // sometimes we get division by zero, due to rounding
                boolean improvement = (entropyDecrease < bestEntropyDecrease);

                if (notNaN && improvement) {
                    double leftThreshold = datapoints.get(currentSplitPosition - 1).getFeatureValue(featureId);
                    double rightThreshold = datapoints.get(currentSplitPosition).getFeatureValue(featureId);
                    double midThreshold = (leftThreshold + rightThreshold) / 2.0;

                    bestEntropyDecrease = entropyDecrease;
                    bestSplitThreshold = midThreshold;
                    bestSplitPosition = currentSplitPosition;
                }
            }

            // update previous value, so that the condition in the above mini while loop will be true again, allowing it to run once more
            previousValue = datapoints.get(currentSplitPosition).getFeatureValue(featureId);
        }
        
        if (bestSplitPosition != null) {
        	
        	List<FeatureVector> leftDatapoints = datapoints.subList(0,  bestSplitPosition); // these are views - no actual copying has happened
        	List<FeatureVector> rightDatapoints = datapoints.subList(bestSplitPosition,  totalSamples);
        	
        	double metricGainFromSplit = bestEntropyDecrease - entropyDecreaseWithoutSplit; // subtract what would have been gained without bothering
        	
        	return new BranchNode(depth, bestSplitThreshold, featureId, metricGainFromSplit, leftDatapoints, rightDatapoints);
        	
        } else {
        	return null; // return null if no split found
        }
        
    }

}
