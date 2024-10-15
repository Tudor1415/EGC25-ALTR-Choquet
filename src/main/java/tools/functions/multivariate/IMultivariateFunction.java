package tools.functions.multivariate;

import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;

public interface IMultivariateFunction {

    double computeScore(IAlternative[] alternatives);

    double computeScore(double score0, double score1);

    double computeScore(DecisionRule[] rules);
    
    String getName();

    void setName(String name);
}