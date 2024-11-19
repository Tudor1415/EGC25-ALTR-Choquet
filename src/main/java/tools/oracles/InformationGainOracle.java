package tools.oracles;

import lombok.Getter;
import tools.rules.DecisionRule;
import tools.functions.singlevariate.InformationGainScoreFunction;

public class InformationGainOracle extends ArtificialOracle {
    @Getter
    public String TYPE = "InformationGain";

    @Getter
    private InformationGainScoreFunction scoreFunction;

    public InformationGainOracle(int nbTransactions) {
        scoreFunction = new InformationGainScoreFunction(nbTransactions);
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return scoreFunction.computeScore(rule);
    }
}