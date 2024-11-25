package tools.functions.multivariate;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;

@Getter
@Setter
public class PairwiseUncertainty implements CertaintyFunction {
    public String Name;

    // The associated out ranking certainty
    private CertaintyFunction theta;
    private IMultivariateFunction regularization = new zeroFunction();

    public PairwiseUncertainty(String name, CertaintyFunction theta, IMultivariateFunction regularization) {
        Name = name;
        this.theta = theta;
        this.regularization = regularization;
    }

    public PairwiseUncertainty(String name, CertaintyFunction theta) {
        Name = name;
        this.theta = theta;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        return 1 - Math.abs(1 - 2 * getTheta().computeScore(alternatives)) + regularization.computeScore(alternatives);
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return 1 - Math.abs(1 - 2 * getTheta().computeScore(rules)) + regularization.computeScore(rules);
    }

    @Override
    public double computeScore(double score0, double score1) {
        return 1 - Math.abs(1 - 2 * getTheta().computeScore(score0, score1));
    }

    @Override
    public void setScoreFunction(ISinglevariateFunction scoreFunction) {
        getTheta().setScoreFunction(scoreFunction);
    }

    @Override
    public ISinglevariateFunction getScoreFunction() {
        return getTheta().getScoreFunction();
    }

}
