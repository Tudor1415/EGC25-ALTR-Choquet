package tools.oracles;

import lombok.Getter;
import tools.functions.singlevariate.OWA.OWALexmin;
import tools.rules.DecisionRule;

@Getter
public class OWAOracle extends ArtificialOracle {

    public String TYPE = "owa";

    private OWALexmin scoreFunction;

    public OWAOracle(double delta, int nbCriteria) {
        scoreFunction = new OWALexmin(delta, nbCriteria);
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return scoreFunction.computeScore(rule);
    }
}