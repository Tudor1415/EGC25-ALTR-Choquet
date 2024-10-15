package tools.utils;

import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

import com.zaxxer.sparsebits.SparseBitSet;

public class SetUtil {

    /**
     * Create a BitSet with the specified size and one bit set at the index
     * posSetBit
     * 
     * @param size      size of the BitSet
     * @param posSetBit index of the bit to set
     * @return the created BitSet
     */
    public static BitSet createBitSet(int size, int posSetBit) {
        BitSet bits = new BitSet(size);
        bits.set(posSetBit);
        return bits;
    }

    /**
     * Checks if a BitSet is a subset of another BitSet.
     *
     * @param b  The first BitSet.
     * @param b2 The second BitSet.
     * @return True if {@code b1} is a subset of {@code b2}, false otherwise.
     */
    public static boolean isSubsetOf(BitSet b1, BitSet b2) {
        // Clone b to avoid modifying the original BitSet
        BitSet cloneB = (BitSet) b1.clone();

        // Perform AND operation with b2 on the cloned BitSet
        cloneB.and(b2);

        // Check if the result is equal to the original BitSet b
        return cloneB.equals(b1);
    }

    /**
     * Converts an integer to a BitSet with the specified initial capacity.
     *
     * @param value    The integer value to convert.
     * @param capacity Initial capacity of the BitSet.
     * @return The corresponding BitSet.
     */
    public static BitSet intToBitSet(int value, int capacity) {
        BitSet bitSet = new BitSet(capacity);

        // Iterate through each bit position and set the corresponding bit in the BitSet
        for (int i = 0; i < capacity; i++) {
            if ((value & (1 << i)) != 0) {
                bitSet.set(i);
            }
        }

        return bitSet;
    }

    /**
     * Computes the union of two sets.
     *
     * @param <T>  the type of elements in the sets
     * @param set1 the first set
     * @param set2 the second set
     * @return a new set containing all distinct elements from both sets
     */
    public static <T> Set<T> union(Set<T> set1, Set<T> set2) {
        Set<T> unionSet = new HashSet<>(set1);
        unionSet.addAll(set2);
        return unionSet;
    }

    /**
     * Creates a new SparseBitSet with all bits set to 1.
     *
     * @param length The length of the SparseBitSet.
     * @return A SparseBitSet with all bits set to 1.
     */
    public static SparseBitSet createSparseBitSetAllOnes(int length) {
        SparseBitSet bitSet = new SparseBitSet(length);
        bitSet.set(0, length);
        bitSet.cardinality(); // Updating the sets statistics
        return bitSet;
    }

    /**
     * Copies the elements of the original cover to a new SparseBitSet.
     * 
     * @param originalCover The original cover to be copied.
     * @return A copy of the original cover.
     */
    public static SparseBitSet copyCover(SparseBitSet originalCover) {
        SparseBitSet copiedCover = new SparseBitSet(originalCover.size());
        copiedCover.or(originalCover); // Copy elements from the original cover
        copiedCover.cardinality(); // Updating the set statistics
        return copiedCover;
    }

    /**
     * Copies the elements of the original set to a new HashSet.
     * 
     * @param originalSet The original set to be copied.
     * @return A copy of the original set.
     */
    public static Set<String> copySet(Set<String> originalSet) {
        return new HashSet<>(originalSet);
    }
}
