package tools.functions.singlevariate.OWA;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

/**
 * Ordered Weighted Average function (OWA)
 * Note that the vector is sorted in increasing order (and not decreasing !)
 */
public class OWAScoreFunction implements ISinglevariateFunction {

    public static String TYPE = "owa";
    public @Setter @Getter String name = "owa";

    private double[] weights;

    public OWAScoreFunction(double[] weights) {
        this.weights = weights;
    }

    @Override
    public double computeScore(IAlternative alternative) {
        assert alternative.getVector().length == weights.length;
        double score = 0;
        for (int i = 0; i < weights.length; i++) {
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
        return computeScore(alternative);
    }
}