package model;

import java.util.List;
import java.util.concurrent.ExecutorService;

// represents a node that has already split
class BranchNode extends AbstractNode {

    // also has "depth" inherited from AbstractNode
    private AbstractNode leftNode;
    private AbstractNode rightNode;
    private double threshold;
    private int splittingFeatureId;
    private double metricGain; // the entropy decrease from children minus from parent

    BranchNode(int depth, double threshold, int splittingFeatureId, double metricGain,
    					List<FeatureVector> leftDatapoints, List<FeatureVector> rightDatapoints) {
        super(depth);
        this.threshold = threshold;
        this.splittingFeatureId = splittingFeatureId;
        this.metricGain = metricGain;

        leftNode = new LeafNode(this.depth + 1, leftDatapoints);
        rightNode = new LeafNode(this.depth + 1, rightDatapoints);
    }
    
    double getMetricGain() {
    	return metricGain;
    }
    
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(super.toString());
        builder.append(", Feature: ");
        builder.append(splittingFeatureId);
        builder.append(", Threshold: ");
        builder.append(String.format("%.4f", threshold));
        builder.append(", Gain: ");
        builder.append(String.format("%.4f",  metricGain));
        builder.append(" {");
        builder.append(leftNode.toString());
        builder.append(" | ");
        builder.append(rightNode.toString());
        builder.append("}");
        return builder.toString();
    }


    // here, we (attempt to) split the children
    // (and if the children split, then we recursively attempt to split the grandchildren)
    @Override
    AbstractNode split(Config config, FeatureSelector selector, ExecutorService exec) {
        leftNode = leftNode.split(config, selector, exec);
        rightNode = rightNode.split(config, selector, exec);

        return this;
    }

    @Override
    void performLogitIncrement(FeatureVector vector) {

        double featureValue = vector.getFeatureValue(splittingFeatureId);

        if (featureValue <= threshold) {
            leftNode.performLogitIncrement(vector);
        }
        else {
            rightNode.performLogitIncrement(vector);
        }
    }
    
    @Override
    void updateFeatureImportances(FeatureImportances importances) {
    	importances.increment(splittingFeatureId, metricGain);
    	leftNode.updateFeatureImportances(importances);
    	rightNode.updateFeatureImportances(importances);
    }

}
