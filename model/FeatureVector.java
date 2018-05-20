package model;

class FeatureVector implements Comparable<FeatureVector> {

    private boolean label;
    private double logit; // will be set incrementally, both in training and in predicting
    private double[] featureValues;
    
    FeatureVector(boolean label, double[] featureValues) {
        this.label = label;
        this.logit = 0.0;
        this.featureValues = featureValues;
    }
    
    boolean getLabel() {
        return label;
    }
    
    double getLogit() {
        return logit;
    }
    
    double getFeatureValue(int featureId) {
        return featureValues[featureId];
    }
    
    void incrementLogit(double deltaLogit) {
    	logit += deltaLogit;
    }
    
    void clearLogit() {
    	logit = 0.0;
    }
    
    double getProb() {
    	// apply logistic function
    	double expLogit = Math.exp(logit);
    	return expLogit / (1.0 + expLogit);
    }
    
    // first derivative of entropy
    double getFirstDeriv() {
    	if (label) {
    		return getProb() - 1.0;
    	} else {
    		return getProb();
    	}
    }
    
    // second derivative of entropy
    double getSecondDeriv() {
    	double prob = getProb();
    	return prob * (1.0 - prob);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < featureValues.length; i++) {
            builder.append(featureValues[i]);
            if (i < featureValues.length - 1)
                builder.append(",");
            else
                builder.append(",");
        }
        builder.append(label ? "1" : "0");
        builder.append(",");
        builder.append(String.format("%.3f",  getProb()));
        return builder.toString();
    }
    
    // for calculating precision-recall curves
    // will sort in *descending* order
    @Override
	public int compareTo(FeatureVector other) {
    	return - Double.compare(this.logit, other.logit);
    }

}
