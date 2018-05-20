package model;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GBTModel {

    private List<AbstractNode> trees; // Each AbstractNode in this list is the **root** of a tree
    private FeatureImportances importances;

    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (AbstractNode tree : trees) {
            builder.append(tree.toString());
            builder.append("\n");
        }
        return builder.toString();
    }
    
    public void printFeatureImportances() {
    	System.out.println(importances);
    }

    // don't use this constructor; instead, use the train method as a factory
    private GBTModel(List<AbstractNode> trees, FeatureImportances importances) {
        this.trees = trees;
        this.importances = importances;
    }

    // a factory - returns a GBTModel object whose trees are fitted to data according to config
    public static GBTModel train(Config config, Data data) {
    	
    	data.clearLogits();
    	
    	FeatureSelector featureSelector = new FeatureSelector(config, data.getNumFeatures());
    	List<AbstractNode> trainedTrees = new ArrayList<>(config.getNumTrees());
    	
    	FeatureImportances importances = new FeatureImportances(data);
    	
    	ExecutorService exec = Executors.newFixedThreadPool(config.getNumThreads());
    	
    	for (int treeId = 0; treeId < config.getNumTrees(); treeId++) {
    		AbstractNode rootNode = new LeafNode(1, data.getFeatureVectors());
    		rootNode = rootNode.split(config, featureSelector, exec);
    		trainedTrees.add(rootNode);
    		rootNode.updateFeatureImportances(importances);
    	}
    	
    	exec.shutdown();
        
        data.markAsFitted();
        
        return new GBTModel(trainedTrees, importances);
    }


    // for scoring an entire test set in batch
    public void predict(Data testData) {
        
    	testData.clearLogits();
    	
    	testData.getFeatureVectors()
    		.parallelStream()
    		.forEach(vector -> {trees.forEach(tree -> {tree.performLogitIncrement(vector);});});
    	
    	testData.markAsFitted();
    }

}
