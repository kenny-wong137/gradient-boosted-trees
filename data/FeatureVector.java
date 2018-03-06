package data;

public class FeatureVector implements Comparable<FeatureVector> {

    private boolean label;
    private double logit; // will be set incrementally, both in training and in predicting
    private double[] featureValues;
    
    public FeatureVector(boolean label, double[] featureValues) {
        this.label = label;
        this.logit = 0.0;
        this.featureValues = featureValues;
    }
    
    public boolean getLabel() { return label; }
    public double getLogit() { return logit; }
    public double getFeatureValue(int featureId) { return featureValues[featureId]; }
    
    public void incrementLogit(double deltaLogit) {
    	logit += deltaLogit;
    }
    
    public void clearLogit() {
    	logit = 0.0;
    }
    
    
    public double getProb() {
    	// apply logistic function
    	double expLogit = Math.exp(logit);
    	return expLogit / (1.0 + expLogit);
    }
    
    // first derivative of entropy
    public double getFirstDeriv() {
    	if (label) {
    		return getProb() - 1.0;
    	} else {
    		return getProb();
    	}
    }
    
    // second derivative of entropy
    public double getSecondDeriv() {
    	double prob = getProb();
    	return prob * (1.0 - prob);
    }

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
