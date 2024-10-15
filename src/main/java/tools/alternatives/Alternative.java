package tools.alternatives;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

import lombok.Getter;
import lombok.Setter;

/**
 * The alternative class is used to efficiently encode an alternative.
 * In this context, the alternative is the set of measures given to
 * the oracle or any other ranking algorithms. We do not work on the
 * rules directly but rather on the alternatives constructed on said
 * rules.
 */
public class Alternative implements IAlternative {

    @Getter
    @Setter
    private double[] vector;

    /**
     * Ordered indexes of x such that x[orderedPermutation[i]] <=
     * x[orderedPermutation[i+1]]
     */
    private int[] orderedPermutation;

    /**
     * Constructs an alternative with the provided measures vector.
     *
     * @param vector The vector representing the alternative.
     */
    public Alternative(double[] vector) {
        this.vector = vector;
    }

    /**
     * Constructs an alternative with a zero-filled measure vector of the given
     * length.
     *
     * @param length The length of the vector.
     */
    public Alternative(int length) {
        this.vector = zerosArray(length);
    }

    /**
     * Creates a new array filled with zeros.
     *
     * @param length The length of the array.
     * @return An array filled with zeros.
     */
    private static double[] zerosArray(int length) {
        return new double[length];
    }

    @Override
    public int[] getOrderedPermutation() {
        if (orderedPermutation == null) {
            orderedPermutation = IntStream
                    .range(0, vector.length)
                    .boxed()
                    .sorted(Comparator.comparingDouble(i -> vector[i]))
                    .mapToInt(i -> i)
                    .toArray();
        }
        return orderedPermutation;
    }

    @Override
    public double getOrderedValue(int i) {
        return vector[getOrderedPermutation()[i]];
    }

    // Deep copy constructor
    public Alternative(Alternative original) {
        // Deep copy the vector array
        this.vector = original.vector.clone();

        // Deep copy the orderedPermutation array if it's not null
        if (original.orderedPermutation != null) {
            this.orderedPermutation = original.orderedPermutation.clone();
        }
    }

    public IAlternative deepCopy() {
        return new Alternative(this.vector.clone());
    }

    @Override
    public int hashCode() {
        final int prime = 499;
        int result = 1;
        result = prime * result + Arrays.hashCode(vector);
        result = prime * result + Arrays.hashCode(orderedPermutation);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Alternative other = (Alternative) obj;
        if (!Arrays.equals(vector, other.vector))
            return false;
        return true;
    }
}
