package tools.utils.kappalab;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import lombok.Getter;
import tools.utils.SetUtil;

/**
 * This class represents Möbius capacities constructed from linear capacities
 * FIXME: Is this the correct term linear?
 */
public class MobiusCapacity {

    // Map to store capacity values for each capacity set
    private Map<BitSet, Double> capacities = new HashMap<>();

    // Number of criteria
    private @Getter int nbCriteria;

    // Additivity parameter for the Mobius capacity
    private @Getter int kAdditivity;

    // Ordered array of capacity sets
    private @Getter BitSet[] orderedCapacitySets;

    /**
     * Constructs a MobiusCapacity instance using a pre-defined capacity map.
     *
     * @param nbCriteria  The number of criteria.
     * @param kAdditivity The additivity parameter for the Mobius capacity.
     * @param capacities  The pre-defined capacities for each capacity set.
     */
    public MobiusCapacity(int nbCriteria, int kAdditivity, Map<BitSet, Double> capacities) {
        this.nbCriteria = nbCriteria;
        this.kAdditivity = kAdditivity;
        this.capacities = capacities;
        orderCapacitySets(capacities.keySet());
    }

    /**
     * Constructs a MobiusCapacity instance using an array of capacity values.
     *
     * @param nbCriteria    The number of criteria.
     * @param kAdditivity   The additivity parameter for the Mobius capacity.
     * @param capacityArray An array of capacity values in the order specified by
     *                      capacity sets.
     */
    public MobiusCapacity(int nbCriteria, int kAdditivity, double[] capacityArray) {
        this.nbCriteria = nbCriteria;
        this.kAdditivity = kAdditivity;

        // Select capacity sets based on additivity constraint
        int maxNbCapacitySets = (int) Math.pow(2, nbCriteria);
        Set<BitSet> selectedCapacitySets = new HashSet<>();
        for (int i = 0; i < maxNbCapacitySets; i++) {
            BitSet capacitySet = SetUtil.intToBitSet(i, nbCriteria);

            // Additivity constraint
            if (capacitySet.cardinality() <= kAdditivity) {
                selectedCapacitySets.add(capacitySet);
            }

        }

        // Order the capacity sets and populate capacities map
        orderCapacitySets(selectedCapacitySets);
        for (int i = 0; i < orderedCapacitySets.length; i++) {
            capacities.put(orderedCapacitySets[i], capacityArray[i]);
        }
    }

    /**
     * Checks if BitSet a is lexicographically before BitSet b.
     *
     * @param a          The first BitSet.
     * @param b          The second BitSet.
     * @param nbCriteria The number of criteria.
     * @return True if a is lexicographically before b; false otherwise.
     */
    private boolean isLexBefore(BitSet a, BitSet b, int nbCriteria) {
        for (int i = 0; i < nbCriteria; i++) {
            if (a.get(i) && !b.get(i)) {
                return true;
            } else if (!a.get(i) && b.get(i)) {
                return false;
            }
        }
        return false;
    }

    /**
     * Orders the given capacity sets lexicographically based on cardinality and
     * lexicographical comparison.
     *
     * @param selectedCapacitySets The set of capacity BitSets to be ordered.
     */
    private void orderCapacitySets(Set<BitSet> selectedCapacitySets) {
        orderedCapacitySets = new ArrayList<>(selectedCapacitySets).stream().sorted((o1, o2) -> {
            if (o1.cardinality() < o2.cardinality()) {
                return -1;
            }
            if (o2.cardinality() < o1.cardinality()) {
                return 1;
            }
            if (isLexBefore(o1, o2, nbCriteria)) {
                return -1;
            }
            if (isLexBefore(o2, o1, nbCriteria)) {
                return 1;
            }
            return 0;
        }).toArray(BitSet[]::new);
    }

    /**
     * Retrieves the capacity value associated with the given capacity BitSet.
     *
     * @param capacitySet The BitSet representing a capacity set.
     * @return The corresponding capacity value.
     */
    public double getCapacityValue(BitSet capacitySet) {
        return capacities.get(capacitySet);
    }

    /**
     * Returns a string representation of the object, displaying capacity sets and
     * their associated values.
     *
     * @return A string containing capacity sets and their values.
     */
    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (BitSet capacitySet : orderedCapacitySets) {
            str.append(capacitySet).append(" ").append(capacities.get(capacitySet)).append("\n");
        }
        return str.toString();
    }

    /**
     * Retrieves an array of capacity values corresponding to the ordered capacity
     * sets.
     *
     * @return An array of capacity values in the order specified by
     *         orderedCapacitySets.
     */
    public double[] getOrderedCapacityValues() {
        return Arrays.stream(orderedCapacitySets).mapToDouble(b -> getCapacityValue(b)).toArray();
    }
}
