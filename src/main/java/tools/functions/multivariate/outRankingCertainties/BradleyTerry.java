package tools.functions.multivariate.outRankingCertainties;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

public class BradleyTerry implements CertaintyFunction {

    public static final String TYPE = "OutRankingCertainty";
    public @Setter @Getter String name = "BradleyTerryOutRanking";

    /** The function of which we want to compute the out-ranking certainty */
    private @Setter ISinglevariateFunction scoreFunction;

    public BradleyTerry(ISinglevariateFunction scoreFunction) {
        this.scoreFunction = scoreFunction;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        // Compute the score for the given alternatives
        double score0 = scoreFunction.computeScore(alternatives[0]);
        double score1 = scoreFunction.computeScore(alternatives[1]);

        return computeScore(score0, score1);
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return computeScore(new IAlternative[] {rules[0].getAlternative(), rules[1].getAlternative()});
    }

    @Override
    public double computeScore(double score0, double score1) {
        return Math.exp(score0) / (Math.exp(score0) + Math.exp(score1));
    }
    
}
