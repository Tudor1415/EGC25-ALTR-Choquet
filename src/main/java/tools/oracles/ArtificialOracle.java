package tools.oracles;

import org.apache.commons.math3.distribution.NormalDistribution;

import lombok.Getter;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;

/**
 * Abstract class representing a score-based oracle for comparing alternatives.
 */
public abstract class ArtificialOracle implements Oracle{
    private @Getter RandomUtil random = new RandomUtil();

    /**
     * Compares two alternatives based on their computed scores.
     *
     * @param a The first alternative to compare.
     * @param b The second alternative to compare.
     * @return A negative integer, zero, or a positive integer if the score of
     *         alternative 'a' is less than, equal to, or greater than the score of
     *         alternative 'b', respectively.
     */
    public int compare(DecisionRule rule_a, DecisionRule rule_b) {
        double scoreA = computeScore(rule_a);
        double scoreB = computeScore(rule_b);

        int decision = -Double.compare(scoreA, scoreB);

        if(scoreA == scoreB)
            System.out.println("Oracle cannot decide !");

        return decision;
    }

    public int compareNoisy(DecisionRule rule_a, DecisionRule rule_b, double noise) {
        double u_a = getScoreFunction().computeScore(rule_a);
        double u_b = getScoreFunction().computeScore(rule_b);

        NormalDistribution normalDistribution = new NormalDistribution(u_b, noise);

        double prob = normalDistribution.cumulativeProbability(u_a);

        return getRandom().Bernoulli(prob) ? 1 : -1;
    }

    /**
     * Computes the score for a given alternative.
     *
     * @param a The alternative for which to compute the score.
     * @return The computed score for the given alternative.
     */
    public abstract double computeScore(DecisionRule rule);

    /**
     * Gets the score function used by the oracle.
     *
     * @return The score function.
     */
    public abstract ISinglevariateFunction getScoreFunction();

    /**
     * Gets the TYPE of the oracle.
     *
     * @return The TYPE of the oracle.
     */
    public abstract String getTYPE();
}
