package tools.oracles;

import lombok.Getter;
import tools.functions.singlevariate.ChiSquaredScoreFunction;
import tools.rules.DecisionRule;

public class ChiSquaredOracle extends ArtificialOracle {
    @Getter
    public String TYPE = "chiSquared";

    @Getter
    private ChiSquaredScoreFunction scoreFunction;

    public ChiSquaredOracle(int nbTransactions) {
        scoreFunction = new ChiSquaredScoreFunction(nbTransactions);
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return scoreFunction.computeScore(rule);
    }
}