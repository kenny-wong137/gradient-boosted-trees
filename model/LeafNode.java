package model;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

// Represents a terminal mode (although this may decide to split later)
class LeafNode extends AbstractNode {

    // also has "depth" inherited from AbstractNode
    private List<FeatureVector> datapoints;
    private Double deltaLogit = null; // will be assigned value when finalised

    LeafNode(int depth, List<FeatureVector> datapoints) {
        super(depth);
        this.datapoints = datapoints;
    }
    
    @Override
    public String toString() {
    	StringBuilder builder = new StringBuilder(super.toString());
    	if (deltaLogit != null) {
    		builder.append(", Boost: " + String.format("%.4f",  deltaLogit));
    	}
    	return builder.toString();
    }
    
    // calculates the deltalogit, then applies this increment to all datapoints
    private void finalise(Config config) {
    	
    	double sumFirstDerivs = datapoints.stream().mapToDouble(vector -> vector.getFirstDeriv()).sum();
    	double sumSecondDerivs = datapoints.stream().mapToDouble(vector -> vector.getSecondDeriv()).sum();
    	deltaLogit = - config.getLearningRate() * sumFirstDerivs / (sumSecondDerivs + config.getL2reg());
    			// Newton-Raphson step
    	
    	datapoints.forEach(vector -> {vector.incrementLogit(deltaLogit);} );
    	
    	datapoints = null; // clears memory

    }

    // Splits as far as possible. Returns reference to the fully-split version of this node.
    @Override
    AbstractNode split(Config config, FeatureSelector selector, ExecutorService exec) {

        boolean tooDeep = (config.getMaxTreeDepth() != null) && (depth >= config.getMaxTreeDepth());

        // if we've already reached the max depth, then we should not do the split
        if (tooDeep) {
        	finalise(config);
            return this;
        }
        
        boolean notEnoughPoints = (datapoints.size() < 2 * config.getMinSamplesLeaf());
        if (notEnoughPoints) {
        	finalise(config);
        	return this;
        }
        
        try {
        	// will now attempt to choose best split (and best splitting feature)
        	List<Integer> featureSelection = selector.sampleFeatures();
        	BranchNode bestSplit = null; // null for the moment; will remain null until we find a valid split
        	Double bestMetricGain = null;

        	List<Split> splittingTasks = new ArrayList<>();
        	for (Integer featureId : featureSelection) {
        		Split task = new Split(config, featureId, datapoints, depth);
        		splittingTasks.add(task);
        	}
        
        	List<Future<BranchNode>> splittingOutcomes = exec.invokeAll(splittingTasks);
        
        	for (Future<BranchNode> outcome : splittingOutcomes) {
        		BranchNode splitUsingThisFeature = outcome.get();
            	if (splitUsingThisFeature != null) {
                	double metricGainWithThisFeature = splitUsingThisFeature.getMetricGain();
                	boolean improvesMetric = (bestMetricGain == null) || (metricGainWithThisFeature < bestMetricGain);
                		// NB the more *negative* the better
                	if (improvesMetric) {
                    	bestSplit = splitUsingThisFeature;
                    	bestMetricGain = metricGainWithThisFeature;
                	}
            	} 
        	}


        	// if we're unable to find any feature with a split that satisfy minSamplesLeaf, then we can't split.
        	if (bestSplit == null) {
        		finalise(config);
            	return this;
        	}

        	// In this final most interesting case where we DO a split, the output is a BranchNode, not a LeafNode
        	return bestSplit.split(config, selector, exec);
        	
        } catch (ExecutionException | InterruptedException ex) {
        	throw new RuntimeException(ex);
        }
    }


    @Override
    void performLogitIncrement(FeatureVector vector) {
        vector.incrementLogit(deltaLogit);
    }
    
    @Override
    void updateFeatureImportances(FeatureImportances importances) {
    	// do nothing
    }

}
