package tools.functions.multivariate.outRankingCertainties;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;

public class ScoreDifference implements CertaintyFunction {

    public static final String TYPE = "OutRankingCertainty";
    public @Setter @Getter String name = "ScoreDifferenceOutRanking";
    public @Setter @Getter double eps = 0.001;

    /** The function of which we want to compute the out-ranking certainty */
    private @Setter ISinglevariateFunction scoreFunction;


    public ScoreDifference(ISinglevariateFunction scoreFunction) {
        this.scoreFunction = scoreFunction;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        // Compute the score for the given alternatives
        double score0 = scoreFunction.computeScore(alternatives[0]);
        double score1 = scoreFunction.computeScore(alternatives[1]);

        return computeScore(score0, score1);
    }

    public double computeScore(DecisionRule[] rules) {
        return computeScore(new IAlternative[] {rules[0].getAlternative(), rules[1].getAlternative()});
    }

    @Override
    public double computeScore(double score0, double score1) {
        return (1-getEps()) * ((score0 - score1) / 2 + 0.5) + getEps();
    }
    
}
