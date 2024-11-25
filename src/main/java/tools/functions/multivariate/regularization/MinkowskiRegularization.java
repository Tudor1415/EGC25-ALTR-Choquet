package tools.functions.multivariate.regularization;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.IMultivariateFunction;

@Setter
@Getter
public class MinkowskiRegularization implements IMultivariateFunction {
    private String Name = "MinkowskiDistance";
    private double p = 2.0; // Default to Euclidean distance (L2 norm)

    public MinkowskiRegularization() {
    }

    public MinkowskiRegularization(double p) {
        if (p <= 0) {
            throw new IllegalArgumentException("The order 'p' must be greater than 0.");
        }
        this.p = p;
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        double[] vector1 = alternatives[0].getVector();
        double[] vector2 = alternatives[1].getVector();

        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors must be of the same length.");
        }

        double sum = 0.0;

        for (int i = 0; i < vector1.length; i++) {
            sum += Math.pow(Math.abs(vector1[i] - vector2[i]), p);
        }

        return Math.pow(sum, 1.0 / p);
    }

    @Override
    public double computeScore(double score0, double score1) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeScore'");
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return computeScore(new IAlternative[] { rules[0].getAlternative(), rules[1].getAlternative() });
    }
}
