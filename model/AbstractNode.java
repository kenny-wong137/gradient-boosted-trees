package model;

import java.util.concurrent.ExecutorService;

abstract class AbstractNode {
	
    int depth;

    AbstractNode(int depth) {
        this.depth = depth;
    }

    @Override
    public String toString() {
        StringBuilder builder  = new StringBuilder();
        builder.append("Depth: ");
        builder.append(depth);
        return builder.toString();
    }

    // Splits this node, and children, and so on, until no further splits are possible
    abstract AbstractNode split(Config config, FeatureSelector selector, ExecutorService exec);
    
    // for prediction (alters the logit attribute within featureVector)
    abstract void performLogitIncrement(FeatureVector featureVector);
    
    abstract void updateFeatureImportances(FeatureImportances importances);
}
