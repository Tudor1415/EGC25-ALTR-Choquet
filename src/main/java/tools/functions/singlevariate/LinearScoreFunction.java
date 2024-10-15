package tools.functions.singlevariate;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;

/**
 * Linear score function (weighted sum of the measures).
 */
public class LinearScoreFunction implements ISinglevariateFunction {

    /**
     * The type identifier for linear score functions.
     */
    public static String TYPE = "linear";
    public @Setter @Getter String name = "linear";

    /**
     * The weights associated with each criterion.
     */
    private double[] weights;

    private boolean weightsInitialized;

    public LinearScoreFunction(double[] weights) {
        this.weights = weights;
    }

    public LinearScoreFunction() {
        this.weightsInitialized = false;
    }

    /**
     * Computes the score for the given alternative based on the linear score
     * function.
     *
     * @param alternative The alternative for which to compute the score.
     * @return The computed score.
     * @throws AssertionError if the length of the alternative's vector does not
     *                        match the length of the weights array.
     */
    @Override
    public double computeScore(IAlternative alternative) {
        double score = 0;
        for (int i = 0; i < alternative.getVector().length; i++) {
            if (!weightsInitialized)
                score += alternative.getVector()[i];
            else
                score += weights[i] * alternative.getVector()[i];
        }
        return score;
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(rule);
    }
}