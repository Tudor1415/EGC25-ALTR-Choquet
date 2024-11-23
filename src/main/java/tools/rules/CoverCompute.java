package tools.rules;

import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import com.zaxxer.sparsebits.SparseBitSet;

import tools.data.Dataset;
import tools.utils.SetUtil;

/**
 * Class for sequential computation of covers.
 */
public class CoverCompute {
    private Dataset dataset;
    private Map<String, SparseBitSet> itemsMap;

    /**
     * Constructor for CoverCompute.
     * 
     * @param dataset  The dataset.
     * @param itemsMap The map of items to SparseBitSet covers.
     */
    public CoverCompute(Dataset dataset) {
        this.dataset = dataset;
        this.itemsMap = dataset.getItemsMap();
    }

    /**
     * Computes the cover sequentially for a given set of items.
     * 
     * @param itemsInSet The set of items.
     * @return The computed cover.
     */
    public SparseBitSet compute(Set<String> itemsInSet) {
        // Get the initial cover based on the first item in the set, or create a new one if none exists
        SparseBitSet finalCover = getInitialCover(itemsInSet);

        // Sequentially AND all other covers with the initial cover
        for (String item : itemsInSet) {
            SparseBitSet cover = itemsMap.get(item);
            if (cover != null) {
                finalCover.and(cover);
            }
        }

        return finalCover;
    }

    /**
     * Retrieves the initial cover from the items map or creates a new empty cover if the set is empty.
     * 
     * @param itemsInSet The set of items.
     * @return A SparseBitSet that will be used as the base for further computation.
     */
    private SparseBitSet getInitialCover(Set<String> itemsInSet) {
        if (itemsInSet.isEmpty()) {
            return new SparseBitSet();
        } else {
            String firstItem = itemsInSet.iterator().next();
            SparseBitSet initialCover = itemsMap.getOrDefault(firstItem, new SparseBitSet());
            return SetUtil.copyCover(initialCover);
        }
    }
}
