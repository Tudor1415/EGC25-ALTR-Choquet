package tools.functions.singlevariate.Choquet;

import java.util.BitSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;
import tools.utils.kappalab.MobiusCapacity;

/**
 * Implementation of the Choquet integral using the Möbius transform for score
 * computation.
 * This class represents a scoring function based on Choquet integral where the
 * capacities are derived from the Möbius transform. The scoring function is
 * computed by combining the minimum values of criteria in selected capacity
 * sets with their corresponding capacities.
 *
 * @param capacity The Möbius capacity containing information about capacity
 *                 sets and their values.
 */
public class ChoquetMobiusScoreFunction implements ISinglevariateFunction {

    /**
     * Type identifier for this score function.
     */
    public static String TYPE = "mobiusChoquet";
    public @Setter @Getter String name = "mobiusChoquet";

    /**
     * The Möbius capacity used in the Choquet scoring function.
     */
    private @Getter MobiusCapacity capacity;

    public ChoquetMobiusScoreFunction(MobiusCapacity capacity) {
        this.capacity = capacity;
    }

    /**
     * Computes the minimum value of an alternative in a specific capacity set.
     *
     * @param capacitySet The capacity set to consider.
     * @param alternative The alternative for which the minimum value is computed.
     * @return The minimum value of the alternative in the specified capacity set.
     */
    private double getMinVal(BitSet capacitySet, IAlternative alternative) {
        double minVal = Double.MAX_VALUE;
        for (int i = capacitySet.nextSetBit(0); i > -1; i = capacitySet.nextSetBit(i + 1)) {
            minVal = Math.min(minVal, alternative.getVector()[i]);
        }
        return minVal;
    }

    /**
     * Computes the Choquet integral score for the given alternative based on Möbius
     * capacities.
     *
     * @param alternative The alternative for which the Choquet score is computed.
     * @return The Choquet score for the alternative.
     */
    @Override
    public double computeScore(IAlternative alternative) {
        double score = 0;
        BitSet[] orderedCapacitySets = capacity.getOrderedCapacitySets();
        for (int i = 1; i < orderedCapacitySets.length; i++) {
            BitSet capacitySet = orderedCapacitySets[i];
            score += getMinVal(capacitySet, alternative) * capacity.getCapacityValue(capacitySet);
        }
        return score;
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(rule);
    }
}