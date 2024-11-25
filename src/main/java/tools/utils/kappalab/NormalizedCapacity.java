package tools.utils.kappalab;

import lombok.Getter;
import tools.utils.SetUtil;

import java.util.Map;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;

public class NormalizedCapacity {
    private Map<BitSet, Double> capacities = new HashMap<>();
    private @Getter int nbCriteria;
    private @Getter int nbCapacitySets;

    public NormalizedCapacity(int nbCriteria) {
        this.nbCriteria = nbCriteria;
        nbCapacitySets = (int) Math.pow(2, nbCriteria);
        capacities.put(SetUtil.intToBitSet(0, nbCriteria), 0.0); // Empty set has capacity 0
        capacities.put(SetUtil.intToBitSet(nbCapacitySets - 1, nbCriteria), 1.0); // Full set has capacity 1
    }

    public NormalizedCapacity(int nbCriteria, double[] weights) {
        this.nbCriteria = nbCriteria;
        nbCapacitySets = (int) Math.pow(2, nbCriteria);
        for (int i = 0; i < nbCapacitySets; i++) {
            capacities.put(SetUtil.intToBitSet(i, nbCriteria), weights[i]);
        }
    }

    public NormalizedCapacity(int nbCriteria, double equalWeight) {
        this(nbCriteria);
        double[] weights = new double[(int) Math.pow(2, nbCriteria) - 2];
        Arrays.fill(weights, equalWeight);
        for (int i = 1; i < weights.length + 1; i++) { // Exclude empty set and full set
            capacities.put(SetUtil.intToBitSet(i, nbCriteria), weights[i - 1]);
        }
    }

    // Constructor to generate random capacities
    public NormalizedCapacity(int nbCriteria, boolean random) {
        this(nbCriteria);
        if (random) {
            generateRandomCapacities();
        }
    }

    private void generateRandomCapacities() {
        BitSet[] subsets = new BitSet[nbCapacitySets];
        for (int i = 0; i < nbCapacitySets; i++) {
            subsets[i] = SetUtil.intToBitSet(i, nbCriteria);
        }

        Arrays.sort(subsets, (a, b) -> Integer.compare(a.cardinality(), b.cardinality())); // Sort by size

        double lastValue = 0.0;
        for (BitSet subset : subsets) {
            if (subset.cardinality() == 0) continue; // Skip empty set
            if (subset.cardinality() == nbCriteria) continue; // Skip full set

            double randomValue = lastValue + Math.random() * (1.0 - lastValue); // Ensure monotonicity
            capacities.put(subset, randomValue);
            lastValue = randomValue;
        }

        capacities.put(SetUtil.intToBitSet(nbCapacitySets - 1, nbCriteria), 1.0); // Full set has capacity 1
    }

    public double getCapacityValue(BitSet capacitySet) {
        return capacities.getOrDefault(capacitySet, 0.0);
    }

    public double getCapacityValue(int capacitySetIndex) {
        return getCapacityValue(SetUtil.intToBitSet(capacitySetIndex, nbCriteria));
    }

    public void addCapacitySet(BitSet capacitySet, double value) {
        assert value >= 0 && value <= 1;
        capacities.put(capacitySet, value);
    }

    public void addCapacitySet(int capacitySetIndex, double value) {
        addCapacitySet(SetUtil.intToBitSet(capacitySetIndex, nbCriteria), value);
    }

    public boolean containsCapacitySet(BitSet capacitySet) {
        return capacities.containsKey(capacitySet);
    }

    public boolean containsCapacitySet(int capacitySetIndex) {
        return containsCapacitySet(SetUtil.intToBitSet(capacitySetIndex, nbCriteria));
    }

    public double[] getWeights() {
        double[] weights = new double[nbCapacitySets];
        for (int i = 0; i < nbCapacitySets; i++) {
            weights[i] = capacities.get(SetUtil.intToBitSet(i, nbCriteria));
        }
        return weights;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < nbCapacitySets; i++) {
            BitSet currentCapacitySet = SetUtil.intToBitSet(i, nbCriteria);
            str.append(currentCapacitySet + " : " + capacities.get(currentCapacitySet) + "\n");
        }
        return str.toString();
    }
}
