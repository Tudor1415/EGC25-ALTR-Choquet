package tools.functions.multivariate.regularization;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.IMultivariateFunction;

@Setter
@Getter
public class CosineRegularization implements IMultivariateFunction {
    private String Name = "CosineDistance";

    @Override
    public double computeScore(IAlternative[] alternatives) {
        double[] vector1 = alternatives[0].getVector();
        double[] vector2 = alternatives[1].getVector();

        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Vectors must be of the same length.");
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
            norm1 += vector1[i] * vector1[i];
            norm2 += vector2[i] * vector2[i];
        }

        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        if (norm1 == 0.0 || norm2 == 0.0) {
            throw new IllegalArgumentException("Vectors must not be zero vectors.");
        }

        double cosineSimilarity = dotProduct / (norm1 * norm2);
        return 1.0 - cosineSimilarity; // Cosine distance is 1 - cosine similarity
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
