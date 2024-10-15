package tools.functions.singlevariate;

import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;

public interface ISinglevariateFunction {

    double computeScore(DecisionRule rule);

    double computeScore(IAlternative alternative);

    double computeScore(IAlternative alternative, DecisionRule rule);

    String getName();

    void setName(String name);
}