package tools.utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;
import tools.rules.IRule;
import tools.rules.RuleMeasures;

public class AlternativeUtil {
    /**
     * This method computes an alternative given a rule.
     * The measures of which the alternative is made of are the measures
     * specified in the parameter measureNames.
     * 
     * @param rule           A rule
     * @param nbTransactions The total number of transactions in the dataset
     * @param smoothCounts   The smoothing factor for the counts
     * @param measureNames   Array of measure names that constitute the alternative
     * @return A set of alternatives from those rules
     */
    public static IAlternative computeAlternative(
            IRule rule,
            int nbTransactions,
            double smoothCounts,
            String[] measureNames) {

        // Compute the value of the measures
        double[] measuresValue = new RuleMeasures(
                rule,
                nbTransactions,
                smoothCounts)
                .computeMeasures(measureNames);

        return new Alternative(measuresValue);
    }

    /**
     * This method computes an alternative given a rule.
     * If the rule has no antecedent or consequent, it returns an alternative with a
     * vector full of zeros.
     * The measures of which the alternative is made of are the measures
     * specified in the parameter measureNames.
     * 
     * @param rule           A rule
     * @param nbTransactions The total number of transactions in the dataset
     * @param smoothCounts   The smoothing factor for the counts
     * @param measureNames   Array of measure names that constitute the alternative
     * @return A set of alternatives from those rules
     */
    public static IAlternative computeAlternativeOrZero(
            IRule rule,
            int nbTransactions,
            double smoothCounts,
            String[] measureNames) {

        if (rule.getItemsInX().isEmpty() || rule.getY().isEmpty()) {
            // In the case of a rule with no antecedent or consequent, initialize
            // the alternative to zero
            return new Alternative(measureNames.length);
        } else {
            // We compute the alternative of the rule
            return computeAlternative(rule, nbTransactions, smoothCounts,
                    measureNames);
        }
    }

    /**
     * This method computes a list of alternatives given a set of rules.
     * The measures of which the alternatives are made of are the measures
     * specified in the parameter measureNames.
     * 
     * @param rules          A set of rules
     * @param nbTransactions The total number of transactions in the dataset
     * @param smoothCounts   The smoothing factor for the counts
     * @param measureNames   Array of measure names that constitute the alternative
     * @return A set of alternatives from those rules
     */
    public static Set<IAlternative> computeAlternatives(
            Set<DecisionRule> rules,
            int nbTransactions,
            double smoothCounts,
            String[] measureNames) {
        Set<IAlternative> alternatives = new HashSet<>(rules.size());

        for (DecisionRule rule : rules) {
            alternatives.add(computeAlternative(rule, nbTransactions, smoothCounts, measureNames));
        }

        return alternatives;
    }

    /**
     * Generates a list of alternatives with equally spaced points in the specified
     * range for each dimension.
     *
     * @param n         The number of equally spaced points.
     * @param minValues An array containing the minimum value for each dimension.
     * @param maxValues An array containing the maximum value for each dimension.
     * @return A list of alternatives.
     */
    public static List<IAlternative> generateEquallySpacedAlternatives(int n, double[] minValues, double[] maxValues) {
        List<IAlternative> alternatives = new ArrayList<>();

        // Calculate spacing between points for each dimension
        double[] spacings = new double[minValues.length];
        for (int i = 0; i < minValues.length; i++) {
            spacings[i] = (maxValues[i] - minValues[i]) / (Math.sqrt(n) - 1);
        }

        // Generate points
        int[] indices = new int[minValues.length];
        while (indices[0] < Math.sqrt(n)) {
            double[] point = new double[minValues.length];
            for (int i = 0; i < minValues.length; i++) {
                point[i] = minValues[i] + indices[i] * spacings[i];
            }
            alternatives.add(new Alternative(point));
            incrementIndices(indices, (int) Math.sqrt(n));
        }

        return alternatives;
    }

    // Helper method to increment indices in a multi-dimensional array
    private static void incrementIndices(int[] indices, int gridSize) {
        for (int i = indices.length - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] >= gridSize && i > 0) {
                indices[i] = 0;
            } else {
                break;
            }
        }
    }
}
