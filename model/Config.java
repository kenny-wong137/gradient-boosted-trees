package model;

public class Config {

    private int minSamplesLeaf = 1;

    private Integer numFeaturesSplit = null; // null means infinity

    private Integer maxTreeDepth = null;

    private double learningRate = 1.0;
    
    private double l2reg = 0.0;
    private static final double minL2reg = 0.01;
    
    private double minGainSplit = 0.0;

    private int numTrees = 1;
    
    private int numThreads = Math.max(Runtime.getRuntime().availableProcessors() - 1, 1);

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Trees: ");
        builder.append(numTrees);
        if (maxTreeDepth != null) {
            builder.append("; Depth: ");
            builder.append(maxTreeDepth);
        }
        if (minSamplesLeaf > 1) {
            builder.append("; Samples leaf: ");
            builder.append(minSamplesLeaf);
        }
        if (numFeaturesSplit != null) {
            builder.append("; Features: ");
            builder.append(numFeaturesSplit);
        }
        if (learningRate != 1.0) {
            builder.append("; Learning rate: ");
            builder.append(learningRate);
        }
        if (l2reg != 0.0) {
        	builder.append("; L2 reg: ");
        	builder.append(l2reg);
        }
        if (minGainSplit != 0.0) {
        	builder.append("; Min Gain: ");
        	builder.append(minGainSplit);
        }
        return builder.toString();
    }

    public int getMinSamplesLeaf() {
        return minSamplesLeaf;
    }
    
    public Integer getNumFeaturesSplit() {
        return numFeaturesSplit;
    }
    
    public Integer getMaxTreeDepth() {
        return maxTreeDepth;
    }
    
    public int getNumTrees() {
        return numTrees;
    }

    public double getLearningRate() {
        return learningRate;
    }
    
    public double getL2reg() {
        return l2reg;
    }
    
    public double getMinGainSplit() {
        return minGainSplit;
    }
    
    public int getNumThreads() {
        return numThreads;
    }


    // *** Define builders. ***

    // mark default constructor as private, forcing everybody to use the builders
    private Config() {}

    public static class Builder {

        protected Config config = new Config(); // carries the default settings

        // if not used, then minSamplesLeaf = 1
        public Builder setMinSamplesLeaf(int minSamplesLeaf) {
            if (minSamplesLeaf >= 1) {
                config.minSamplesLeaf = minSamplesLeaf;
            }
            else {
                throw new IllegalArgumentException("Cannot set min samples leaf below 1.");
            }
            return this;
        }

        // if not used, then numFeaturesSplit = null (so we use all the available features)
        public Builder setNumFeaturesSplit(int numFeaturesSplit) {
            if (numFeaturesSplit >= 1) {
                config.numFeaturesSplit = numFeaturesSplit;
            }
            else {
                throw new IllegalArgumentException("Cannot set num features split less than 1.");
            }
            return this;
        }

        // if not used, then maxTreeDepth = null (so we grow the trees to the bottom)
        public Builder setMaxTreeDepth(int maxTreeDepth) {
            if (maxTreeDepth >= 1) {
                config.maxTreeDepth = maxTreeDepth;
            }
            else {
                throw new IllegalArgumentException("Cannot set max tree depth less than 1.");
            }
            return this;
        }

        // if not used, then learning rate is 1
        public Builder setLearningRate(double learningRate) {
            if (learningRate > 0.0 && learningRate <= 1.0) {
                config.learningRate = learningRate;
            }
            else {
                throw new IllegalArgumentException("Learning rate must be between 0 and 1.");
            }
            return this;
        }


        // if not used, then numTrees = 1
        public Builder setNumTrees(int numTrees) {
            if (numTrees >= 1) {
                config.numTrees = numTrees;
            }
            else {
                throw new IllegalArgumentException("Cannot set num trees below 1.");
            }
            return this;
        }
        
        // if not used, then zero
        public Builder setL2reg(double l2reg) {
        	if (l2reg >= minL2reg) {
        		config.l2reg = l2reg;
        	} else {
        		throw new IllegalArgumentException("L2 reg must be at least " + minL2reg);
        	}
        	return this;
        }
        
        // if not used, then zero
        public Builder setMinGainSplit(double minGainSplit) {
        	if (minGainSplit >= 0.0) {
        		config.minGainSplit = minGainSplit;
        	} else {
        		throw new IllegalArgumentException("Min gain split must be non-negative.");
        	}
        	return this;
        }
        
        public Builder setNumThreads(int numThreads) {
        	if (numThreads >= 1) {
        		config.numThreads = numThreads;
        	} else {
        		throw new IllegalArgumentException("Must have at least one thread.");
        	}
        	return this;
        }
        
        public Config build() {
            return config;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

}
