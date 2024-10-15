package tools.utils;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

public class SamplingUtil {

    public static double calculateEntropy(Collection<Integer> ruleSampleFrequencies, int totalSamples) {
        double entropy = 0.0;

        for (int frequency : ruleSampleFrequencies) {
            double probability = (double) frequency / totalSamples;
            if (probability != 0) {
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }

        return entropy;
    }

    /**
     * Computes the probability repartition function of a certain rule metric.
     * 
     * @param ruleOccurrences  The list of number of occurrences for each rule in
     *                         the
     *                         sample.
     * @param ruleMetricValues The list of corresponding rule scores.
     * @return The percentage of samples that are in %k of the best score.
     */
    public static HashMap<Double, Double> computeDistribution(List<Integer> ruleOccurrences,
            List<Double> ruleMetricValues) {
        double sumOfFrequencies = ruleOccurrences.stream()
                .mapToInt(Integer::intValue)
                .sum();

        double[][] pairs = IntStream.range(0, ruleOccurrences.size())
                .mapToObj(i -> new double[] { ruleMetricValues.get(i), ruleOccurrences.get(i) })
                .sorted(Comparator.comparingDouble((double[] pair) -> pair[0]))
                .toArray(double[][]::new);

        /*
         * Maps the score to the percentage of rules that have a score below the aid
         * score
         */
        HashMap<Double, Double> scoreCumulativeDistribution = new HashMap<>();
        double lastPercentage = 0;
        for (int idx = 0; idx < ruleMetricValues.size(); idx++) {
            lastPercentage += pairs[idx][1] / sumOfFrequencies;
            scoreCumulativeDistribution.put(pairs[idx][0], lastPercentage);
        }

        return scoreCumulativeDistribution;
    }

}
