package tools.utils.kappalab;

import lombok.Getter;
import tools.utils.SetUtil;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

public class NormalizedCapacity {

    private Map<BitSet, Double> capacities = new HashMap<>();
    private @Getter int nbCriteria;
    private @Getter int nbCapacitySets;

    public NormalizedCapacity(int nbCriteria) {
        this.nbCriteria = nbCriteria;
        nbCapacitySets = (int) Math.pow(2, nbCriteria);
        capacities.put(SetUtil.intToBitSet(0, nbCriteria), 0.0);
        capacities.put(SetUtil.intToBitSet(nbCapacitySets - 1, nbCriteria), 1.0);
    }

    public NormalizedCapacity(int nbCriteria, double[] weights) {
        this.nbCriteria = nbCriteria;
        nbCapacitySets = (int) Math.pow(2, nbCriteria);
        for (int i = 0; i < nbCapacitySets; i++) {
            capacities.put(SetUtil.intToBitSet(i, nbCriteria), weights[i]);
        }
    }

    // In NormalizedCapacity class
    public NormalizedCapacity(int nbCriteria, double equalWeight) {
        this.nbCriteria = nbCriteria;
        nbCapacitySets = (int) Math.pow(2, nbCriteria) - 1; // Exclude the empty set from counting
        double[] weights = new double[nbCapacitySets];
        Arrays.fill(weights, equalWeight);
        for (int i = 1; i < nbCapacitySets; i++) { // Start from 1 to exclude the empty set
            capacities.put(SetUtil.intToBitSet(i, nbCriteria), weights[i - 1]);
        }
    }

    public double getCapacityValue(BitSet capacitySet) {
        return capacities.get(capacitySet);
    }

    public double getCapacityValue(int capacitySetIndex) {
        return capacities.get(SetUtil.intToBitSet(capacitySetIndex, nbCriteria));
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

    public String toString() {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < nbCapacitySets; i++) {
            BitSet currentCapacitySet = SetUtil.intToBitSet(i, nbCriteria);
            str.append(currentCapacitySet + " : " + capacities.get(currentCapacitySet) + "\n");
        }
        return str.toString();
    }

}
