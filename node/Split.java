package node;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;

import data.FeatureVector;
import model.Config;

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
		
		datapoints = new ArrayList<>(datapoints);  // make shallow copy,
		// because (i) we are accessing datapoints from multiple threads
		// ... and (ii) because datapoints might be a view of a sublist
		// doing this inside the thread
    	
        int totalSamples = datapoints.size();
        
        // already checked in LeafNode class, but just in case...
        if (totalSamples < 2 * config.getMinSamplesLeaf()) {
        	return null;
        }
         
        // sort the samples in the leaf by the value of the chosen feature
        datapoints.sort(Comparator.comparing(vector -> vector.getFeatureValue(featureId)));
        
        // initially, everything except the first minSamplesLeaf datapoints are sent to the right
        double sumLeftFirstDerivs = 0.0;
        double sumLeftSecondDerivs = 0.0;
        double sumRightFirstDerivs = 0.0;
        double sumRightSecondDerivs = 0.0;
        
        for (int position = 0; position < totalSamples; position++) {
        	if (position < config.getMinSamplesLeaf()) {
        		sumLeftFirstDerivs += datapoints.get(position).getFirstDeriv(); // don't use .stream() - too slow
        		sumLeftSecondDerivs += datapoints.get(position).getSecondDeriv();
        	} else {
        		sumRightFirstDerivs += datapoints.get(position).getFirstDeriv(); 
        		sumRightSecondDerivs += datapoints.get(position).getSecondDeriv();
        	}
        }
        
        int currentPosition = config.getMinSamplesLeaf();
        	// NB currentPosition will always be equal to the number of samples sent to the LEFT.
        	// it will also be the index of the datapoint to the right of the split.
        
        double sumAllFirstDerivs = sumLeftFirstDerivs + sumRightFirstDerivs;
        double sumAllSecondDerivs = sumLeftSecondDerivs + sumRightSecondDerivs;

        double entropyDecreaseWithoutSplit
        			= - 0.5 * sumAllFirstDerivs * sumAllFirstDerivs / (sumAllSecondDerivs + config.getL2reg());
        double bestEntropyDecrease = entropyDecreaseWithoutSplit - config.getMinGainSplit();
        	// this is the benchmark to beat
        Double bestSplitThreshold = null;
        Integer bestSplitPosition = null;


        while(true) {
        	
        	double valueToLeft = datapoints.get(currentPosition - 1).getFeatureValue(featureId);
        	double valueToRight = datapoints.get(currentPosition).getFeatureValue(featureId);
        	
        	if (valueToLeft < valueToRight) {
                // Calculate metric gain if splitting here
                double leftEntropyDecrease =
                		- 0.5 * sumLeftFirstDerivs * sumLeftFirstDerivs / (sumLeftSecondDerivs + config.getL2reg());
                double rightEntropyDecrease =
                		- 0.5 * sumRightFirstDerivs * sumRightFirstDerivs / (sumRightSecondDerivs + config.getL2reg());
                double entropyDecrease = leftEntropyDecrease + rightEntropyDecrease;

                if (entropyDecrease < bestEntropyDecrease) {
                    bestEntropyDecrease = entropyDecrease;
                    bestSplitThreshold = (valueToLeft + valueToRight) / 2.0;
                    bestSplitPosition = currentPosition;
                }
        	}
        	
        	// now transfer datapoints across
        	if (currentPosition < totalSamples - config.getMinSamplesLeaf()) {
        		double currentFirstDeriv = datapoints.get(currentPosition).getFirstDeriv();
        		double currentSecondDeriv = datapoints.get(currentPosition).getSecondDeriv();
                
        		sumLeftFirstDerivs += currentFirstDeriv;
        		sumRightFirstDerivs -= currentFirstDeriv;
        		sumLeftSecondDerivs += currentSecondDeriv;
        		sumRightSecondDerivs -= currentSecondDeriv;

        		currentPosition ++;
        		
        	} else {
        		break;
        	}
        }
        
        
        if (bestSplitPosition != null) {
        	
        	List<FeatureVector> leftDatapoints = datapoints.subList(0,  bestSplitPosition);
        	// these are views - no actual copying has happened
        	List<FeatureVector> rightDatapoints = datapoints.subList(bestSplitPosition,  totalSamples);
        	
        	double metricGainFromSplit = bestEntropyDecrease - entropyDecreaseWithoutSplit;
        	// subtract what would have been gained without splitting
        	
        	return new BranchNode(depth, bestSplitThreshold, featureId, metricGainFromSplit,
        									leftDatapoints, rightDatapoints);
        	
        } else {
        	return null; // return null if no split found
        }
        
    }

}
