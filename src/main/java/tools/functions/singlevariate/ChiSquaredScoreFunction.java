package tools.functions.singlevariate;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;
import tools.rules.RuleMeasures;

public class ChiSquaredScoreFunction implements ISinglevariateFunction {
    public static String TYPE = "ChiSquared";

    public @Setter @Getter String name = "ChiSquared";

    private int nbTransactions;

    public ChiSquaredScoreFunction(int nbTransactions) {
        this.nbTransactions = nbTransactions;
    }

    @Override
    public double computeScore(IAlternative alternative) {
        throw new UnsupportedOperationException("Only computes score from decision rule");
    }    

    @Override
    public double computeScore(DecisionRule rule) {
        double phi =  new RuleMeasures(rule, nbTransactions, 1e-6d).computeMeasures(new String[]{RuleMeasures.phi})[0];
        return phi;
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(rule);
    }
}