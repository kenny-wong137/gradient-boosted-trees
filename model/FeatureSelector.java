package model;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

// helper class - for column subsampling
class FeatureSelector {

    private Random generator = new Random();
    private int numFeaturesToSelect;
    private int numFeaturesAvailable;

    FeatureSelector(Config config, int numFeaturesAvailable) {

        if (config.getNumFeaturesSplit() == null) {
            // in this case, there is no limit
            this.numFeaturesToSelect = numFeaturesAvailable;
        } else {
            this.numFeaturesToSelect = Math.min(config.getNumFeaturesSplit(), numFeaturesAvailable);
        }

        this.numFeaturesAvailable = numFeaturesAvailable;
    }

    List<Integer> sampleFeatures() {

        // This calls the random generator more times than strictly necessary,
        // but it looks tidy, and experiments show that it's not a bottleneck.
        List<Integer> featureIdsToShuffle = IntStream.range(0, numFeaturesAvailable).boxed()
                .collect(Collectors.toList());
        Collections.shuffle(featureIdsToShuffle, generator);

        return featureIdsToShuffle.subList(0, numFeaturesToSelect);
    }

}
