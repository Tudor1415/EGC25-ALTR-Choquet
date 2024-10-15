package tools.rules;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntConsumer;

import com.zaxxer.sparsebits.SparseBitSet;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.utils.AlternativeUtil;
import tools.utils.SetUtil;

/**
 * <p>
 * This is a generic decision rule class. An example of a rule would be:
 * </p>
 * <p>
 * X = {64, 12, 15} - This means that the antecedent contains the items in the
 * dataset with values 64, 12, and 15.
 * </p>
 * <p>
 * Y = {2} - This means that the consequent contains the item with value 2 in
 * the
 * dataset.
 * </p>
 *
 * @param x       The antecedent of the rule.
 * @param y       The consequent of the rule.
 * @param dataset The transactional dataset as an array of items.
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class DecisionRule implements IRule {
    // Transactional dataset
    private Dataset dataset;
    private Map<String, SparseBitSet> itemsMap;

    // Frequency variables for X, Y, and Z and smoothing factor
    private @Setter int freqX, freqY, freqZ;
    private @Setter @Getter double smoothCounts;

    // Variables regarding feature vectors computation
    private @Setter @Getter String[] measureNames;
    private @Getter IAlternative alternative;

    // Covers for X, Y, and Z
    private SparseBitSet coverX;
    private SparseBitSet coverY;
    private SparseBitSet coverZ;

    // Memoization maps for covers of X and Z
    private Map<Set<String>, SparseBitSet> memoizedCoverX;
    private Map<Set<String>, SparseBitSet> memoizedCoverZ;
    private @Getter @Setter int maxSizeX, maxSizeZ;

    // Object for parallel cover computation
    private CoverParallelCompute coverComputer;

    // Variable representing the consequent
    private String Y;

    // Set of items in X and Z respectively
    private Set<String> itemsInX;
    private Set<String> itemsInZ;

    public DecisionRule(Set<String> itemsInX, String Y, Dataset dataset, int maxSizeX, int maxSizeZ,
            double smoothCounts, String[] measureNames) {
        // Initializing the transactional dataset
        this.dataset = dataset;
        this.itemsMap = this.dataset.getItemsMap();
        this.smoothCounts = smoothCounts;

        // Initializing all the item sets
        this.itemsInX = itemsInX;
        this.Y = Y;
        computeItemsInZ();

        // Initializing the memoized covers
        setMaxSizeX(maxSizeX);
        setMaxSizeZ(maxSizeZ);
        initializeMemoization(getMaxSizeX(), getMaxSizeZ());

        // Initializing the parallel computer
        this.coverComputer = new CoverParallelCompute(getDataset());

        // Initializing the covers for the antecedent, consequent and their union
        computeNewCover(new String[] { "x", "y", "z" });

        // Initializing all the atomic frequencies
        updateFrequencies(new String[] { "x", "y", "z" });

        // Computing the corresponding feature vector
        setMeasureNames(measureNames);
        this.alternative = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
    }

    public void expandSimpleCopy(DecisionRule originalRule) {
        // Initializing the transactional dataset
        this.dataset = originalRule.getDataset();
        this.itemsMap = getDataset().getItemsMap();

        // Initializing all the item sets
        computeItemsInZ();

        // Initializing the memoized covers
        setMaxSizeX(originalRule.getMaxSizeX());
        setMaxSizeZ(originalRule.getMaxSizeZ());
        initializeMemoization(getMaxSizeX(), getMaxSizeZ());

        // Initializing the parallel computer
        this.coverComputer = new CoverParallelCompute(getDataset());

        // Initializing the covers for the antecedent, consequent and their union
        computeNewCover(new String[] { "x", "y", "z" });

        // Initializing all the atomic frequencies
        updateFrequencies(new String[] { "x", "y", "z" });

        // Computing the corresponding feature vector
        setMeasureNames(originalRule.getMeasureNames());
        this.alternative = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
    }

    /**
     * Initializes memoization for coverX and coverZ with the specified maximum
     * sizes.
     *
     * @param maxSizeX Maximum size for memoizedCoverX
     * @param maxSizeZ Maximum size for memoizedCoverZ
     */
    private void initializeMemoization(int maxSizeX, int maxSizeZ) {
        // Initializing memoizedCoverX with a maximum size of maxSizeX
        this.memoizedCoverX = new LinkedHashMap<Set<String>, SparseBitSet>(maxSizeX, 1.0f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Set<String>, SparseBitSet> eldest) {
                return size() > maxSizeX;
            }
        };

        // Initializing memoizedCoverZ with a maximum size of maxSizeZ
        this.memoizedCoverZ = new LinkedHashMap<Set<String>, SparseBitSet>(maxSizeZ, 1.0f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Set<String>, SparseBitSet> eldest) {
                return size() > maxSizeZ;
            }
        };
    }

    /**
     * Computes the union of sets X and Y and updates the set Z.
     * <p>
     * The union set Z contains all distinct elements present in sets X and Y.
     * </p>
     */
    private void computeItemsInZ() {
        this.itemsInZ = new HashSet<>(this.itemsInX);
        if (!this.Y.isEmpty()) {
            this.itemsInZ.add(this.Y);
        }
    }

    /**
     * Updates the frequencies based on the specified directions.
     *
     * @param directions An array of strings representing the directions for which
     *                   frequencies should be updated.
     */
    private void updateFrequencies(String[] directions) {
        for (String direction : directions) {
            switch (direction) {
                case "x":
                    updateFrequency(this.itemsInX, this.coverX, this::setFreqX);
                    break;
                case "y":
                    updateFrequency(this.Y, this.coverY, this::setFreqY);
                    break;
                case "z":
                    updateFrequency(this.itemsInZ, this.coverZ, this::setFreqZ);
                    break;
            }
        }
    }

    /**
     * Updates the frequency based on the provided items and their corresponding
     * cover.
     * 
     * @param items           The set of items to consider for frequency
     *                        calculation.
     * @param cover           The BitSet representing the cover for the items.
     * @param frequencySetter A consumer to set the calculated frequency.
     */
    private void updateFrequency(Set<?> items, SparseBitSet cover, IntConsumer frequencySetter) {
        int cardinality = cover.cardinality();
        frequencySetter.accept(items.isEmpty() ? 0 : cardinality);
    }

    /**
     * Updates the frequency based on the provided item and its corresponding cover.
     *
     * @param item            The single item to consider for frequency calculation.
     * @param cover           The BitSet representing the cover for the item.
     * @param frequencySetter A consumer to set the calculated frequency.
     */
    private void updateFrequency(String item, SparseBitSet cover, IntConsumer frequencySetter) {
        frequencySetter.accept(item.isEmpty() ? 0 : cover.cardinality());
    }

    /** Updates the rule's feature vector in \mathcal{D} after each change */
    private void updateAlternative() {
        IAlternative modified = AlternativeUtil.computeAlternativeOrZero(this, getDataset().getNbTransactions(),
                smoothCounts, getMeasureNames());
        this.alternative = modified;
    }

    /**
     * Computes covers from present items based on the specified directions.
     *
     * @param directions An array of strings representing the directions for which
     *                   frequencies should be updated.
     */
    private void computeNewCover(String[] directions) {
        for (String direction : directions) {
            switch (direction) {
                case "x":
                    updateCoverX();
                    break;
                case "y":
                    updateCoverY();
                    break;
                case "z":
                    updateCoverZ();
                    break;
                default:
                    throw new IllegalArgumentException("Unexpected direction: " + direction);
            }
        }
    }

    /**
     * Updates coverX based on the items in itemsInX.
     */
    private void updateCoverX() {
        if (!this.itemsInX.isEmpty()) {
            SparseBitSet fromMemory = memoizedCoverX.get(this.itemsInX);
            if (fromMemory != null) {
                this.coverX = SetUtil.copyCover(fromMemory);
            } else {
                // Compute cover in parallel
                this.coverX = coverComputer.compute(itemsInX);
                memoizedCoverX.put(SetUtil.copySet(this.itemsInX), SetUtil.copyCover(this.coverX));
            }
        } else {
            this.coverX = SetUtil.createSparseBitSetAllOnes(this.dataset.getNbTransactions());
        }

    }

    /**
     * Updates coverY based on the items in Y.
     */
    private void updateCoverY() {
        if (!this.Y.isEmpty()) {
            SparseBitSet mapCover = this.itemsMap.get(this.Y);
            this.coverY = (mapCover != null) ? mapCover : new SparseBitSet();
        }
    }

    /**
     * Updates coverZ based on the items in itemsInZ.
     */
    private void updateCoverZ() {
        if (!this.itemsInZ.isEmpty()) {
            SparseBitSet fromMemory = memoizedCoverZ.get(this.itemsInZ);
            if (fromMemory != null) {
                this.coverZ = SetUtil.copyCover(fromMemory);
            } else {
                // Compute cover in parallel
                this.coverZ = coverComputer.compute(itemsInZ);
                memoizedCoverZ.put(SetUtil.copySet(this.itemsInZ), SetUtil.copyCover(this.coverZ));
            }
        } else {
            this.coverZ = SetUtil.createSparseBitSetAllOnes(this.dataset.getNbTransactions());
        }
    }

    /**
     * Sets the value of set X and updates set Z and frequencies accordingly.
     *
     * @param X The new set X to be set.
     */
    public void setX(Set<String> itemsInX) {
        // Sets the items in X and Z accordingly
        this.itemsInX = itemsInX;
        computeItemsInZ();

        // Updates covers and frequencies for directions X and Z
        computeNewCover(new String[] { "x", "z" });
        updateFrequencies(new String[] { "x", "z" });
        updateAlternative();
    }

    /**
     * Adds a value to set X and updates set Z and frequencies accordingly.
     *
     * @param itemValue The item index to be added to set X.
     */
    public void addToX(String itemValue) {
        // Add the item to the antecedent cover and item set
        this.itemsInX.add(itemValue);
        SparseBitSet fromMemoryX = memoizedCoverX.get(this.itemsInX);
        if (fromMemoryX != null) {
            this.coverX = SetUtil.copyCover(fromMemoryX);
        } else {
            // Compute cover using and operation
            this.coverX.and(this.itemsMap.get(itemValue));
            memoizedCoverX.put(SetUtil.copySet(this.itemsInX), SetUtil.copyCover(this.coverX));
        }

        // Compute the new union
        computeItemsInZ();
        SparseBitSet fromMemoryZ = memoizedCoverZ.get(this.itemsInZ);
        if (fromMemoryZ != null) {
            this.coverZ = SetUtil.copyCover(fromMemoryZ);
        } else {
            // Compute cover using and operation
            this.coverZ.and(this.itemsMap.get(itemValue));
            memoizedCoverZ.put(SetUtil.copySet(this.itemsInZ), SetUtil.copyCover(this.coverZ));
        }

        // Update frequencies for sets X and Z
        updateFrequencies(new String[] { "x", "z" });
        updateAlternative();
    }

    /**
     * Removes an item index from set X and updates set Z and frequencies
     * accordingly.
     *
     * @param itemValue The itemValue to be removed from set X.
     * @throws RuntimeException If the specified itemValue is not present in set X.
     */
    public void removeFromX(String itemValue) {
        // Check if the itemValue is present in item set X
        if (this.itemsInX.contains(itemValue)) {
            // Remove the index from X and Z
            this.itemsInX.remove(itemValue);
            computeItemsInZ();

            // Update covers and frequencies for directions X and Z
            computeNewCover(new String[] { "x", "z" });
            updateFrequencies(new String[] { "x", "z" });
            updateAlternative();
        } else {
            // Throw a RuntimeException if the itemValue is not in set X
            throw new RuntimeException("Item index " + itemValue + " not found in item set X.");
        }
    }

    /**
     * Sets the value of set Y and updates set Z and frequencies accordingly.
     *
     * @param Y The new set Y to be set.
     */
    public void setY(String Y) {
        // Sets the items in Y and Z accordingly
        this.Y = Y;
        computeItemsInZ();

        // Updates covers and frequencies for directions Y and Z
        computeNewCover(new String[] { "y", "z" });
        updateFrequencies(new String[] { "y", "z" });
        updateAlternative();
    }

    /**
     * @param obj
     * @return boolean
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        DecisionRule rule = (DecisionRule) obj;
    
        // First, check if itemsInX and Y are equal
        if (Objects.equals(itemsInX, rule.itemsInX) && Objects.equals(Y, rule.Y)) {
            return true;
        }
        
        // Second, check if frequencies are equal
        return Objects.equals(freqX, rule.getFreqX()) &&
               Objects.equals(freqY, rule.getFreqY()) &&
               Objects.equals(freqZ, rule.getFreqZ());
    }    

    @Override
    public int hashCode() {
        return Objects.hash(itemsInX, Y, freqX, freqY, freqZ);
    }

}