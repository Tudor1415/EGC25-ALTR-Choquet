package tools.functions.multivariate.outRankingCertainties;

import org.apache.commons.math3.distribution.NormalDistribution;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

public class Thurstone implements CertaintyFunction {

    public @Setter @Getter String name = "ThurstoneOutRanking";

    /** The function of which we want to compute the out-ranking certainty */
    private @Setter ISinglevariateFunction scoreFunction;

    public Thurstone(ISinglevariateFunction scoreFunction) {
        this.scoreFunction = scoreFunction;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        // Compute the score for the given alternatives
        double score0 = scoreFunction.computeScore(alternatives[0]);
        double score1 = scoreFunction.computeScore(alternatives[1]);

        NormalDistribution normalDistribution = new NormalDistribution(score1, 1d);

        return normalDistribution.cumulativeProbability(score0);
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return computeScore(new IAlternative[] { rules[0].getAlternative(), rules[1].getAlternative() });
    }

    @Override
    public double computeScore(double score0, double score1) {
        NormalDistribution normalDistribution = new NormalDistribution(score1, 1d);

        return normalDistribution.cumulativeProbability(score0);
    }

}
