package tools.functions.singlevariate;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.rules.RuleMeasures;
import tools.alternatives.IAlternative;

public class InformationGainScoreFunction implements ISinglevariateFunction {
    public static String TYPE = "Information Gain";

    public @Setter @Getter String name = "Information Gain";

    private int nbTransactions;

    public InformationGainScoreFunction(int nbTransactions) {
        this.nbTransactions = nbTransactions;
    }

    @Override
    public double computeScore(IAlternative alternative) {
        throw new UnsupportedOperationException("Only computes score from decision rule");
    }    

    @Override
    public double computeScore(DecisionRule rule) {
        double IG =  new RuleMeasures(rule, nbTransactions, 1e-6d).computeMeasures(new String[]{RuleMeasures.informationGain})[0];
        return IG;
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(rule);
    }
}