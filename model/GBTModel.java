package model;
import data.Data;
import data.FeatureSelector;
import node.AbstractNode;
import node.LeafNode;

import java.util.ArrayList;
import java.util.List;

public class GBTModel {

    private List<AbstractNode> trees; // Each AbstractNode in this list is the **root** of a (fully-trained) decision tree

    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (AbstractNode tree : trees) {
            builder.append(tree.toString());
            builder.append("\n");
        }
        return builder.toString();
    }

    // don't use this constructor; instead, use the train method as a factory
    private GBTModel(List<AbstractNode> trees) {
        this.trees = trees;
    }

    // a factory - returns an RFModel object whose trees are fitted to the data according to the config
    public static GBTModel train(Config config, Data data) {
    	
    	data.clearLogits();
    	
    	FeatureSelector featureSelector = new FeatureSelector(config, data.getNumFeatures());
    	List<AbstractNode> trainedTrees = new ArrayList<>(config.getNumTrees());
    	
    	for (int treeId = 0; treeId < config.getNumTrees(); treeId++) {
    		AbstractNode rootNode = new LeafNode(1, data.getFeatureVectors());
    		rootNode = rootNode.split(config, featureSelector);
    		trainedTrees.add(rootNode);
    	}
        
        data.markAsFitted();
        
        return new GBTModel(trainedTrees);

    }


    // for scoring an entire test set in batch
    public void predict(Data testData) {
        
    	testData.clearLogits();
    	
    	testData.getFeatureVectors()
    		.parallelStream()
    		.forEach(vector -> {  trees.forEach(tree -> {tree.performLogitIncrement(vector);}); });
    	
    	testData.markAsFitted();

    }


}
