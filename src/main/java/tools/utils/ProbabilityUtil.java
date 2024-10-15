package tools.utils;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ProbabilityUtil {
    /**
     * Computes the probability distribution of a list of integer values using softmax temperature.
     *
     * @param qValues      The list of integer values.
     * @param temperature The temperature parameter for softmax.
     * @return An array containing the probabilities for each distinct value.
     */
    public static double[] computeSoftmaxDistribution(List<Double> qValues, double temperature) {
        // Find the maximum value
        double maxValue = qValues.stream().mapToDouble(Double::doubleValue).max().orElse(0);

        // Compute exponential of each value divided by temperature
        double[] exponentials = qValues.stream()
                .mapToDouble(value -> Math.exp((value - maxValue) / temperature))
                .toArray();

        // Calculate the sum of exponentials
        double sumOfExponentials = 0;
        for (double exponential : exponentials) {
            sumOfExponentials += exponential;
        }

        // Calculate probabilities using softmax formula
        double[] probabilities = new double[qValues.size()];
        for (int i = 0; i < qValues.size(); i++) {
            probabilities[i] = exponentials[i] / sumOfExponentials;
        }

        return probabilities;
    }

    /**
     * Computes the probability distribution of a list of integer values.
     *
     * @param values The list of integer values.
     * @return An array containing the probabilities for each distinct value.
     */
    public static double[] computeDistribution(List<Integer> values) {
        // Count occurrences of each value
        Map<Integer, Long> valueCounts = values.stream()
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

        // Calculate total count of all values
        long totalCount = values.size();

        // Calculate probabilities for each value
        double[] probabilities = valueCounts.entrySet().stream()
                .mapToDouble(entry -> (double) entry.getValue() / totalCount)
                .toArray();

        return probabilities;
    }

    /**
     * Chooses an index based on the given probability distribution.
     *
     * @param probabilities The probability distribution.
     * @return The chosen index.
     */
    public static int chooseGivenDistribution(double[] probabilities) {
        // Generate a random value between 0 (inclusive) and 1 (exclusive)
        double randomValue = ThreadLocalRandom.current().nextDouble();

        // Accumulate probabilities to find the chosen index
        double cumulativeProbability = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulativeProbability += probabilities[i];
            if (randomValue < cumulativeProbability) {
                return i; // Return the index where the cumulative probability exceeds the random value
            }
        }

        // If no index is chosen yet, return invalid index
        return -1;
    }

    /**
     * Divides each value in the array by the sum of all the values.
     *
     * @param values The array of values to be normalized.
     * @return An array where each value is divided by the sum of all the values.
     */
    public static double[] normalize(double[] values) {
        // Calculate the sum of all values using streams
        double sum = Arrays.stream(values).sum();

        // Normalize the values using streams
        return Arrays.stream(values)
                .map(value -> value / sum)
                .toArray();
    }
}
