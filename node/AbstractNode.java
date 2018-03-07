package node;

import data.FeatureImportances;
import data.FeatureSelector;
import data.FeatureVector;
import model.Config;

public abstract class AbstractNode {
	
    protected int depth;

    public AbstractNode(int depth) {
        this.depth = depth;
    }

    public String toString() {
        StringBuilder builder  = new StringBuilder();
        builder.append("Depth: ");
        builder.append(depth);
        return builder.toString();
    }

    // Splits this node, and children, and so on, until no further splits are possible
    abstract public AbstractNode split(Config config, FeatureSelector selector);
    
    // for prediction (leaves the state of this object alone, and alters the logit attribute within featureVector)
    abstract public void performLogitIncrement(FeatureVector featureVector);
    
    abstract public void updateFeatureImportances(FeatureImportances importances);
}
