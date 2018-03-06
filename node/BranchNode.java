package node;

import java.util.List;

import data.FeatureSelector;
import data.FeatureVector;
import model.Config;

// represents a node that has already split
public class BranchNode extends AbstractNode {

    // also has "depth" inherited from AbstractNode
    private AbstractNode leftNode;
    private AbstractNode rightNode;
    private double threshold;
    private int splittingFeatureId;
    private double metricGain; // the entropy decrease from children minus from parent
    	// (although in reality this would only be achieved if we do the full step, not multiplied by the learning rate)


    public BranchNode(int depth, double threshold, int splittingFeatureId, double metricGain,
    					List<FeatureVector> leftDatapoints, List<FeatureVector> rightDatapoints) {
        super(depth);
        this.threshold = threshold;
        this.splittingFeatureId = splittingFeatureId;
        this.metricGain = metricGain;

        leftNode = new LeafNode(this.depth + 1, leftDatapoints);
        rightNode = new LeafNode(this.depth + 1, rightDatapoints);
    }
    
    public double getMetricGain() {
    	return metricGain;
    }

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


    // here, we (attempt to) split the children (and if the children split, then we recursively attempt to split the grandchildren)
    @Override
    public AbstractNode split(Config config, FeatureSelector selector) {
        leftNode = leftNode.split(config, selector);
        rightNode = rightNode.split(config, selector);

        return this;
    }

    @Override
    public void performLogitIncrement(FeatureVector vector) {

        double featureValue = vector.getFeatureValue(splittingFeatureId);

        if (featureValue <= threshold) {
            leftNode.performLogitIncrement(vector);
        }
        else {
            rightNode.performLogitIncrement(vector);
        }
    }

}
