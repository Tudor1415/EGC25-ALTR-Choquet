package tools.rules;

import static java.lang.Math.*;

/**
 * Class to compute different measures for a rule (inspired by R code : <a href=
 * "https://github.com/mhahsler/arules/blob/master/R/interestMeasures.R">arules</a>)
 *
 * @param rule           The rule for which measures are computed.
 * @param nbTransactions The total number of transactions in the dataset.
 * @param smoothCounts   The smoothing factor for the counts.
 */
public class RuleMeasures {

    // For rounding errors
    public static double epsilon = 1d;

    private double n;
    private double n11;
    private double n1x;
    private double nx1;
    private double n0x;
    private double nx0;
    private double n10;
    private double n01;
    private double n00;

    // Measure names
    public static final String confidence = "confidence";
    public static final String lift = "lift";
    public static final String cosine = "cosine";
    public static final String phi = "phi";
    public static final String kruskal = "kruskal";
    public static final String yuleQ = "yuleQ";
    public static final String addedValue = "pavillon";
    public static final String certainty = "certainty";
    public static final String support = "support";
    public static final String revsupport = "revsup";

    public RuleMeasures(IRule rule, int nbTransactions, double smoothCounts) {
        if (rule == null) {
            throw new RuntimeException("Rule must not be null");
        }
        
        n = nbTransactions;
        n11 = rule.getFreqZ();
        n1x = rule.getFreqX();
        nx1 = rule.getFreqY();
        n0x = n - n1x; // Frequency of transactions without antecedent (X)
        nx0 = n - nx1; // Frequency of transactions without consequent (Y)
        n10 = n1x - n11; // Frequency of transactions with antecedent but without consequent
        n01 = nx1 - n11; // Frequency of transactions with consequent but without antecedent
        n00 = n0x - n01; // Frequency of transactions without both antecedent and consequent

        if (smoothCounts > 0) {
            n = n + 4 * smoothCounts;
            n11 = n11 + smoothCounts;
            n10 = n10 + smoothCounts;
            n01 = n01 + smoothCounts;
            n00 = n00 + smoothCounts;
            n0x = n0x + 2 * smoothCounts;
            nx0 = nx0 + 2 * smoothCounts;
            n1x = n1x + 2 * smoothCounts;
            nx1 = nx1 + 2 * smoothCounts;
        }
    }

    /**
     * Checks if the given value is within the specified range and throws a
     * RuntimeException if not.
     *
     * @param value       The value to be checked.
     * @param lb          The lower bound of the valid range.
     * @param ub          The upper bound of the valid range.
     * @param measureName The name of the measure being checked, used in the error
     *                    message.
     * @throws RuntimeException If the value is outside the valid range.
     */
    private void checkMeasure(double value, double lb, double ub, String measureName) {
        if (value > (ub + epsilon) || value < (lb - epsilon)) {
            throw new IllegalArgumentException("Illegal value for measure " + measureName + 
                ": value=" + value + ", should be between " + lb + " and " + ub);
        }
    }
    

    /**
     * Computes the confidence measure for the rule.
     *
     * @return The confidence measure.
     */
    private double confidence() {
        double value = n11 / n1x;
        checkMeasure(value, 0, 1, confidence);
        return value;
    }

    /**
     * Computes the lift measure for the rule.
     *
     * @return The lift measure.
     */
    private double lift() {
        double value = n * n11 / (n1x * nx1);
        checkMeasure(value, 0, Double.MAX_VALUE, lift);
        return value;
    }

    /**
     * Computes the cosine measure for the rule.
     *
     * @return The cosine measure.
     */
    private double cosine() {
        double value = n11 / sqrt(n1x * nx1);
        checkMeasure(value, 0, 1, cosine);
        return value;
    }

    /**
     * Computes the phi measure for the rule.
     *
     * @return The phi measure.
     */
    private double phi() {
        double value = (n * n11 - n1x * nx1) / sqrt(n1x * nx1 * n0x * nx0);
        checkMeasure(value, -1, 1, phi);
        return value;
    }

    /**
     * Computes the kruskal measure for the rule.
     *
     * @return The kruskal measure.
     */
    private double kruskal() {
        double max_x0x1 = max(nx1, nx0);
        double value = (max(n11, n10) + max(n01, n00) - max_x0x1) / (n - max_x0x1);
        checkMeasure(value, 0, 1, kruskal);
        return value;
    }

    /**
     * Computes the OR measure for the rule.
     *
     * @return The OR measure.
     */
    private double OR() {
        return n11 * n00 / (n10 * n01);
    }

    /**
     * Computes the yuleQ measure for the rule.
     *
     * @return The yuleQ measure.
     */
    private double yuleQ() {
        double OR = OR();
        double value = (OR - 1) / (OR + 1);
        checkMeasure(value, -1, 1, yuleQ);
        return value;
    }

    /**
     * Computes the added value measure for the rule.
     *
     * @return The added value measure.
     */
    private double addedValue() {
        double value = n11 / n1x - nx1 / n;
        checkMeasure(value, -0.5, 1, addedValue);
        return value; 
    }

    /**
     * Computes the certainty measure for the rule.
     *
     * @return The certainty measure.
     */
    private double certainty() {
        double value1 = (n11 / n1x - nx1 / n) / (1 - nx1 / n);
        double value2 = (n11 / nx1 - n1x / n) / (1 - n1x / n);
        double value = max(value1, value2);
        checkMeasure(value, -1, 1, certainty);
        return value;
    }

    /**
     * Computes the support measure for the rule.
     *
     * @return The support measure.
     */
    private double support() {
        double value = n11 / n;
        checkMeasure(value, 0, 1, support);
        return value;
    }

    /**
     * Computes the revsupport measure for the rule.
     *
     * @return The revsupport measure.
     */
    private double revsupport() {
        double value = 1 - support();
        checkMeasure(value, 0, 1, revsupport);
        return value;
    }

    /**
     * Computes multiple measures for the rule based on the provided measure names.
     *
     * @param measureNames An array of measure names for which measures should be
     *                     computed.
     * @return An array of computed measures corresponding to the given measure
     *         names.
     * @throws RuntimeException If an unknown measure name is encountered.
     */
    public double[] computeMeasures(String[] measureNames) {
        double[] measures = new double[measureNames.length];
        for (int i = 0; i < measureNames.length; i++) {
            measures[i] = compute(measureNames[i]);
        }
        return measures;
    }

    /**
     * Computes the specified measure for the rule.
     *
     * @param measureName The name of the measure to be computed.
     * @return The computed value of the specified measure.
     * @throws RuntimeException If an unknown measure name is encountered.
     */
    private double compute(String measureName) {
        if (measureName.equals(confidence))
            return confidence();
        if (measureName.equals(lift))
            return lift();
        if (measureName.equals(cosine))
            return cosine();
        if (measureName.equals(phi))
            return phi();
        if (measureName.equals(kruskal))
            return kruskal();
        if (measureName.equals(yuleQ))
            return yuleQ();
        if (measureName.equals(addedValue))
            return addedValue();
        if (measureName.equals(certainty))
            return certainty();
        if (measureName.equals(support))
            return support();
        if (measureName.equals(revsupport))
            return revsupport();
        throw new RuntimeException("This measure doesn't exist : " + measureName);
    }

}
