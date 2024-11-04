package tools.optimization;

import java.util.BitSet;
import java.util.List;
import java.util.stream.Collectors;

import tools.functions.singlevariate.Choquet.ChoquetMobiusScoreFunction;
import tools.rules.DecisionRule;

/**
 * A utility class for computing loss functions, including mean squared error
 * and regularization using L1 norm, for optimization problems involving
 * Choquet integrals and decision rules.
 */
public class lossFunctions {

    /**
     * Checks if the size of the input and output value lists are the same.
     *
     * @param inputValues  The list of input values to be checked.
     * @param outputValues The list of output values to be checked.
     * @throws IllegalArgumentException if the sizes of the input and output value
     *                                  lists do not match.
     */
    public static void checkArguments(List<?> inputValues, List<?> outputValues) {
        if (inputValues.size() != outputValues.size()) {
            throw new IllegalArgumentException("Input and output value lists must have the same size.");
        }
    }

    /**
     * Computes the mean squared error between the predicted scores from the model
     * and the actual output values.
     *
     * @param model        The {@link ChoquetMobiusScoreFunction} model used to
     *                     compute scores for the input values.
     * @param inputValues  The list of decision rules used as input for the model.
     * @param outputValues The list of actual output values to compare against.
     * @return The mean squared error between the predicted and actual output
     *         values.
     */
    public static double mean_squared_error(ChoquetMobiusScoreFunction model, List<DecisionRule> inputValues,
                                            List<Double> outputValues) {
        checkArguments(inputValues, outputValues);

        List<Double> predictedOutputs = inputValues.stream()
                .parallel()
                .map(model::computeScore)
                .collect(Collectors.toList());

        double mse = 0;
        for (int i = 0; i < outputValues.size(); i++) {
            double error = outputValues.get(i) - predictedOutputs.get(i);
            mse += error * error;
        }
        return mse / outputValues.size();
    }

    /**
     * Computes the L1 regularization term for the given
     * {@link ChoquetMobiusScoreFunction} model.
     *
     * @param model The {@link ChoquetMobiusScoreFunction} model whose capacity values
     *              are used for computing the L1 norm.
     * @return The L1 norm of the capacity values in the model.
     */
    public static double regularization(ChoquetMobiusScoreFunction model) {
        double l1Norm = 0;

        BitSet[] orderedCapacitySets = model.getCapacity().getOrderedCapacitySets();

        // Compute the L1 norm of the capacity values
        for (BitSet capacitySet : orderedCapacitySets) {
            double capacityValue = model.getCapacity().getCapacityValue(capacitySet);
            l1Norm += Math.abs(capacityValue);
        }

        return l1Norm;
    }
}
