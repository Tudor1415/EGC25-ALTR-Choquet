package tools.utils.kappalab;

import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

/**
 * Represents the output of the Kappalab Choquet integral computation.
 */
@Getter
@Setter
@ToString
public class KappalabOutput {
    private double[] capacities;
    private String[] errorMessages;
    private double[] shapleyValues;
    private double[][] interactionIndices;
    private double[] obj;
}