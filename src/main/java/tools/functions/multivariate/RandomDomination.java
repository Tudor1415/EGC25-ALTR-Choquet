package tools.functions.multivariate;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;

import java.util.Random;

@Getter
@Setter
public class RandomDomination implements IMultivariateFunction {
    private String Name = "RandomDomination";
    private Random random = new Random();
    private double[] dominationVector; // Persisted domination vector

    public RandomDomination(int length) {
        randomizeDomination(length);
    }

    @Override
    public double computeScore(IAlternative[] alternatives) {
        if (alternatives.length != 2) {
            throw new IllegalArgumentException("Exactly two alternatives are required.");
        }

        double[] vector1 = alternatives[0].getVector();
        double[] vector2 = alternatives[1].getVector();

        if (vector1.length != vector2.length) {
            throw new IllegalArgumentException("Both vectors must have the same length.");
        }

        // Ensure the domination vector is initialized
        if (dominationVector == null || dominationVector.length != vector1.length) {
            throw new IllegalStateException("Domination vector is not initialized or has incorrect length.");
        }

        // Compute the difference vector
        double[] differenceVector = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            differenceVector[i] = vector1[i] - vector2[i];
        }

        // Compute the scalar product between domination vector and difference vector
        double score = 0.0;
        for (int i = 0; i < dominationVector.length; i++) {
            score += dominationVector[i] * differenceVector[i];
        }

        return score;
    }

    @Override
    public double computeScore(double score0, double score1) {
        throw new UnsupportedOperationException("Unimplemented method 'computeScore'");
    }

    @Override
    public double computeScore(DecisionRule[] rules) {
        return computeScore(new IAlternative[]{rules[0].getAlternative(), rules[1].getAlternative()});
    }

    /**
     * Public method to generate a random domination vector with values between -1 and 1.
     * The generated vector will persist for future computations.
     *
     * @param length The length of the domination vector.
     */
    public void randomizeDomination(int length) {
        this.dominationVector = new double[length];
        for (int i = 0; i < length; i++) {
            this.dominationVector[i] = -1 + (2 * random.nextDouble()); // Random value between -1 and 1
        }
    }
}
