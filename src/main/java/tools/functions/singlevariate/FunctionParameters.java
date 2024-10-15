package tools.functions.singlevariate;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * Class to save the parameters of a score function in a file
 */
@Getter
@Setter
@ToString
public class FunctionParameters {

    // Type of the function linear, X2 or choquet
    private String functionType;

    // Weights associated with the function
    private double[] weights;

    // Additivity parameter 'k' for the Choquet integral
    private int kAdditivity;

    // Number of criteria (different measures) considered in the function
    private int nbCriteria;

    // Time taken for learning the function to compute
    private double timeToLearn;

    // Flag indicating whether the function learning process timed out
    private boolean timeOut;

    // Error messages related to the function learning process
    private String[] errorMessages;

    // Number of iterations performed for the function learning
    private int nbIterations;

    // Shapley values associated with each criteria
    private double[] shapleyValues;

    // TODO: Explain better
    private double[][] interactionIndices;

    // TODO: Explain better
    private double[] obj;
}