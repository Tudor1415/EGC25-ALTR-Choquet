package tools.functions.multivariate.outRankingCertainties;

import static java.lang.Math.abs;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;

@Getter
@Setter
public class Sensitivity implements CertaintyFunction {

    private String Name;
    private ISinglevariateFunction scoreFunction;

    public Sensitivity(String name, ISinglevariateFunction scoreFunction) {
        Name = name;
        this.scoreFunction = scoreFunction;
    }

    /**
     * Measures the gap between two alternatives based on their score function
     * values and vector differences.
     *
     * @param a The first alternative.
     * @param b The second alternative.
     * @return The computed gap between the two alternatives.
     */
    private double measureGap(IAlternative a, IAlternative b) {
        double num = abs(scoreFunction.computeScore(a) - scoreFunction.computeScore(b));
        double denom = 0d;
        for (int i = 0; i < a.getVector().length; i++) {
            denom += abs(a.getVector()[i] - b.getVector()[i]);
        }
        return num / denom;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        return measureGap(alternatives[0], alternatives[1]);
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return measureGap(rules[0].getAlternative(), rules[1].getAlternative());
    }

    @Override
    public double computeScore(double score0, double score1) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeScore'");
    }

}
