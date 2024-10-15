package tools.utils;

import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.Choquet.ChoquetMobiusScoreFunction;
import tools.utils.kappalab.MobiusCapacity;

/**
 * Utility class for handling functions.
 */
public class FunctionUtil {
    /**
     * Gets a score function based on the provided parameters.
     *
     * @param params The function parameters.
     * @return The corresponding score function.
     * @throws RuntimeException if an invalid or unsupported function type is
     *                          provided.
     */
    public static ISinglevariateFunction getScoreFunction(FunctionParameters params) {
        if (params.getFunctionType().equals(ChoquetMobiusScoreFunction.TYPE)) {
            return new ChoquetMobiusScoreFunction(
                    new MobiusCapacity(params.getNbCriteria(), params.getKAdditivity(), params.getWeights()));
        }
        return null;
    }

    /**
     * Creates and returns a new instance of {@link FunctionParameters} with the
     * specified parameters.
     * This method is a convenient way to construct FunctionParameters objects with
     * the given values.
     *
     * @param functionType The type of the score function, e.g., linear or choquet.
     * @param nbCriteria   The number of criteria in the decision-making context.
     * @param kAdditivity  The additivity constraint for Choquet capacities.
     * @param weights      The weights used in the score function.
     * @param timeToLearn  The time allocated for the learning process.
     * @return A new instance of FunctionParameters initialized with the provided
     *         values.
     */
    public static FunctionParameters getFunctionParameters(String functionType, int nbCriteria,
            int kAdditivity, double[] weights, double timeToLearn) {
        FunctionParameters params = new FunctionParameters();
        params.setFunctionType(functionType);
        params.setWeights(weights);
        params.setNbCriteria(nbCriteria);
        params.setKAdditivity(kAdditivity);
        params.setTimeToLearn(timeToLearn);
        return params;
    }

    /**
     * Creates a FunctionParameters object with the specified error messages.
     *
     * @param errorMessages The array of error messages to be logged.
     * @return FunctionParameters object containing the error messages.
     */
    public static FunctionParameters logErrorFunction(String[] errorMessages) {
        FunctionParameters params = new FunctionParameters();
        params.setErrorMessages(errorMessages);
        return params;
    }

    public static IAlternative minMaxNormalize(IAlternative alternative, IAlternative ideal, IAlternative nadir) {
        double[] normalizedAlternative = new double[alternative.getVector().length];

        // Perform min-max normalization for each attribute
        for (int i = 0; i < alternative.getVector().length; i++) {
            double max = ideal.getVector()[i];
            double min = nadir.getVector()[i];
            double value = alternative.getVector()[i];

            // Perform min-max normalization formula
            normalizedAlternative[i] = min == max ? value : (value - min) / (max - min);
        }

        return new Alternative(normalizedAlternative);
    }
}
