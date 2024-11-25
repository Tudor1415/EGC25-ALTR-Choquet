package tools.functions.multivariate;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;

public class zeroFunction implements IMultivariateFunction {

    private @Getter @Setter String Name = "Zero";

    @Override
    public double computeScore(IAlternative[] alternatives) {
        return 0.0d;
    }

    @Override
    public double computeScore(double score0, double score1) {
        return 0.0d;
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return 0.0d;
    }
}
