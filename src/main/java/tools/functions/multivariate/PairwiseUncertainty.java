package tools.functions.multivariate;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

public class PairwiseUncertainty implements CertaintyFunction{
    public @Getter @Setter String Name;

    // The associated out ranking certainty
    private @Setter @Getter CertaintyFunction theta;

    public PairwiseUncertainty(String name, CertaintyFunction theta) {
        Name = name;
        this.theta = theta;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        return 1 - Math.abs(1 - 2*getTheta().computeScore(alternatives));
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return 1 - Math.abs(1 - 2*getTheta().computeScore(rules));
    }

    @Override
    public double computeScore(double score0, double score1) {
        return 1 - Math.abs(1 - 2*getTheta().computeScore(score0, score1));
    }

    @Override
    public void setScoreFunction(ISinglevariateFunction scoreFunction) {
        getTheta().setScoreFunction(scoreFunction);
    }
    
}
