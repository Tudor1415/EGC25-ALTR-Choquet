package tools.rules;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import com.zaxxer.sparsebits.SparseBitSet;

import tools.data.Dataset;
import tools.utils.SetUtil;

/**
 * Class for parallel computation of covers.
 */
public class CoverParallelCompute {
    private Dataset dataset;
    private Map<String, SparseBitSet> itemsMap;

    private Set<SparseBitSet> coversToCompute;
    /**
     * Constructor for CoverParallelCompute.
     * 
     * @param dataset  The dataset.
     * @param itemsMap The map of items to SparseBitSet covers.
     */
    public CoverParallelCompute(Dataset dataset) {
        this.dataset = dataset;
        this.itemsMap = dataset.getItemsMap();
    }

    /**
     * Computes the cover in parallel for a given set of items.
     * 
     * @param itemsInSet The set of items.
     * @return The computed cover.
     */
    public SparseBitSet compute(Set<String> itemsInSet) {
        // Copy Items in Set Covers
        this.coversToCompute = copyItemsInSetCovers(itemsInSet);

        // Divide and Conquer Loop
        divideAndConquer(this.coversToCompute);

        // Return Final Cover
        return returnFinalCover(this.coversToCompute);
    }

    /**
     * Copies the covers of items in the given set.
     * 
     * @param itemsInSet The set of items.
     * @return The set of copied covers.
     */
    Set<SparseBitSet> copyItemsInSetCovers(Set<String> itemsInSet) {
        Set<SparseBitSet> coversToCompute = new HashSet<>();
        for (String itemValue : itemsInSet) {
            SparseBitSet originalCover = itemsMap.get(itemValue);
            SparseBitSet copiedCover = (originalCover != null) ? SetUtil.copyCover(originalCover) : new SparseBitSet();
            coversToCompute.add(copiedCover);
        }
        return coversToCompute;
    }

    /**
     * Divides and conquers the computed covers until only one cover remains.
     *
     * @param coversToCompute The set of computed covers.
     */
    void divideAndConquer(Set<SparseBitSet> coversToCompute) {
        while (this.coversToCompute.size() > 1) {
            // Pair covers
            Set<SparseBitSet[]> coverPairs = pairCovers(this.coversToCompute);

            // Compute covers in parallel
            Set<SparseBitSet> newCoversToCompute = computeCoversParallel(coverPairs);

            // Update the computed covers for the next iteration
            this.coversToCompute = newCoversToCompute;
        }
    }

    /**
     * Computes covers in parallel for the given cover pairs.
     *
     * @param coverPairs The set of cover pairs.
     * @return A set containing newly computed covers.
     */
    Set<SparseBitSet> computeCoversParallel(Set<SparseBitSet[]> coverPairs) {
        // Set to store newly computed covers
        Set<SparseBitSet> coversToCompute = new HashSet<>();

        coverPairs.parallelStream().forEach(pair -> {
            // Compute the bitwise AND operation for the pair
            SparseBitSet newCover = computeAndOperation(pair);

            synchronized (coversToCompute) {
                // Add the computed cover to the set
                coversToCompute.add(newCover);
            }
        });

        return coversToCompute;
    }

    /**
     * Pair the covers in the provided set.
     *
     * @param coversToCompute The set of covers to pair.
     * @return A set containing pairs of covers.
     */
    Set<SparseBitSet[]> pairCovers(Set<SparseBitSet> coversToCompute) {
        Set<SparseBitSet[]> coverPairs = new HashSet<>();
        Iterator<SparseBitSet> iterator = coversToCompute.iterator();

        while (iterator.hasNext()) {
            SparseBitSet cover1 = iterator.next();
            if (iterator.hasNext()) {
                SparseBitSet cover2 = iterator.next();
                coverPairs.add(new SparseBitSet[] { cover1, cover2 });
            } else {
                coverPairs.add(
                        new SparseBitSet[] { cover1, SetUtil.createSparseBitSetAllOnes(dataset.getNbTransactions()) });
            }
        }

        return coverPairs;
    }

    /**
     * Returns the final cover from the computed covers set.
     * 
     * @param coversToCompute The set of computed covers.
     * @return The final cover.
     */
    SparseBitSet returnFinalCover(Set<SparseBitSet> coversToCompute) {
        return coversToCompute.iterator().next();
    }

    /**
     * Computes the bitwise AND operation between two covers.
     * 
     * @param pair The pair of covers.
     * @return The result of the AND operation.
     */
    SparseBitSet computeAndOperation(SparseBitSet[] pair) {
        SparseBitSet cover1 = pair[0];
        SparseBitSet cover2 = pair[1];
        cover1.and(cover2); // Perform bitwise AND operation in place
        return cover1;
    }
}
