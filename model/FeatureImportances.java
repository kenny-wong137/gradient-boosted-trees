package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class FeatureImportances {

    private List<SingleImportance> importances;

    FeatureImportances(Data data) {
        importances = new ArrayList<>(data.getNumFeatures());
        for (int featureId = 0; featureId < data.getNumFeatures(); featureId++) {
            importances.add(new SingleImportance(data.getFeatureName(featureId)));
        }
    }

    void increment(int featureId, double extraGain) {
        importances.get(featureId).increment(extraGain);
    }

    @Override
    public String toString() {
        List<SingleImportance> importancesToSort = new ArrayList<>(importances);
        Collections.sort(importancesToSort);
        StringBuilder builder = new StringBuilder("\nImportances:\n");
        for (SingleImportance importance : importancesToSort) {
            builder.append(importance);
        }
        return builder.toString();
    }

    private class SingleImportance implements Comparable<SingleImportance> {

        String name;
        double metricGain;

        SingleImportance(String name) {
            this.name = name;
            this.metricGain = 0.0;
        }

        void increment(double extraGain) {
            metricGain += extraGain;
        }

        @Override
        public String toString() {
            return String.format("%.4f", metricGain) + " : " + name + "\n";
        }

        @Override
        public int compareTo(SingleImportance other) {
            return Double.compare(this.metricGain, other.metricGain);
        }
    }
    
}
