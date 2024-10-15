package tools.functions.singlevariate.OWA;

import static java.lang.Math.pow;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

/**
 * Implementation of Lexmin using OWA operator
 * See Yager - On the analytical representation of the Leximin ordering and its
 * application to flexible constraint propagation
 */
public class OWALexmin implements ISinglevariateFunction {
    public static String TYPE = "OWALexmin";
    public @Setter @Getter String name = "OWALexmin";

    private OWAScoreFunction func;

    public OWALexmin(double delta, int nbCriteria) {
        double[] weights = new double[nbCriteria];
        for (int i = 0; i < nbCriteria - 1; i++) {
            weights[i] = pow(delta, i) / pow(1 + delta, i + 1);
        }
        int i = nbCriteria - 1;
        weights[i] = pow(delta, i) / pow(1 + delta, i);
        func = new OWAScoreFunction(weights);
    }

    @Override
    public double computeScore(IAlternative a) {
        return func.computeScore(a);
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(alternative);
    }
}