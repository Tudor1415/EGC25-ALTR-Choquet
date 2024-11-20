package tools.functions.multivariate;

import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;

public interface CertaintyFunction extends IMultivariateFunction{

    String TYPE = "OutRankingCertainty";

    void setName(String name);

    String getName();

    void setScoreFunction(ISinglevariateFunction scoreFunction);

    ISinglevariateFunction getScoreFunction();

    double computeScore(IAlternative[] alternatives);

    double computeScore(DecisionRule[] rules);

    double computeScore(double score0, double score1);

}