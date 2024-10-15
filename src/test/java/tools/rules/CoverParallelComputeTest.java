package tools.rules;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import com.zaxxer.sparsebits.SparseBitSet;

import tools.data.Dataset;
import tools.utils.SetUtil;
import tools.utils.TestUtils;

import static org.junit.jupiter.api.Assertions.*;

class CoverParallelComputeTest {
    private CoverParallelCompute coverParallelCompute;
    private Dataset dataset;
    private String[][] transactions;
    private Map<String, SparseBitSet> itemsMap;

    @BeforeEach
    public void setUp() throws IOException {
        // Create dataset and items map for testing
        Set<String> classItemValues = new HashSet<>(Arrays.asList("young", "adult", "senior"));
        dataset = new Dataset("groceries_test_file.dat", "src/test/resources/", classItemValues);
        itemsMap = dataset.getItemsMap();
        transactions = dataset.getTransactionalDataset();

        // Initialize CoverParallelCompute instance
        coverParallelCompute = new CoverParallelCompute(dataset);
    }

    /**
     * Tests the {@link CoverParallelCompute#compute(Set)} method for different
     * sizes of input sets.
     * 
     * @throws IOException
     *
     * @see CoverParallelCompute#compute(Set)
     */
    @Test
    public void testCompute() {
        // Items to test the cover compute function for
        String[] items = { "milk", "eggs", "butter", "cheese", "adult" };

        for (int i = 1; i <= items.length; i++) {
            Set<String> itemsInSet = new HashSet<>();
            for (int j = 0; j < i; j++) {
                itemsInSet.add(items[j]);
            }

            SparseBitSet cover = coverParallelCompute.compute(itemsInSet);
            assertNotNull(cover);

            // Perform assertions on the computed cover
            int expected = TestUtils.countTransactionsWithItems(transactions, itemsInSet.toArray(new String[0]));
            assertEquals(expected, cover.cardinality());
        }
    }

    @Test
    public void testComputeIrisBug() throws IOException {
        // Retrieving the iris dataset
        Set<String> classItemValues = new HashSet<>(Arrays.asList("13", "14", "15"));
        Dataset irisDataset = new Dataset("iris.dat", "src/test/resources/", classItemValues);
        String[][] irisTransactions = irisDataset.getTransactionalDataset();

        // Initiating computing instance
        coverParallelCompute = new CoverParallelCompute(irisDataset);

        // Items to test the cover compute function for
        String[] items = { "2", "5", "8", "14" };

        for (int i = 4; i <= items.length; i++) {
            Set<String> itemsInSet = new HashSet<>();
            for (int j = 0; j < i; j++) {
                itemsInSet.add(items[j]);
            }

            SparseBitSet cover = coverParallelCompute.compute(itemsInSet);
            assertNotNull(cover);

            // Perform assertions on the computed cover
            int expected = TestUtils.countTransactionsWithItems(irisTransactions, itemsInSet.toArray(new String[0]));
            assertEquals(expected, cover.cardinality());
        }
    }

    /**
     * Test the {@link CoverParallelCompute#copyItemsInSetCovers(Set)} method of the
     * {@link CoverParallelCompute} class.
     * This method tests that the copied covers have the correct size for each item
     * in the input set.
     */
    @Test
    public void testCopyItemsInSetCovers() {
        Set<String> itemsInSet = new HashSet<>();
        itemsInSet.add("milk");
        itemsInSet.add("eggs");
        Set<SparseBitSet> copiedCovers = coverParallelCompute.copyItemsInSetCovers(itemsInSet);

        assertNotNull(copiedCovers);

        // Perform assertions on the copied covers
        assertEquals(2, copiedCovers.size());

        // Check the size of each cover, we assume that the set preserves the order
        // which is incorrect
        int i = 0;
        for (SparseBitSet itemCover : copiedCovers) {
            int expected = TestUtils.countTransactionsWithItems(transactions, itemsInSet.toArray(new String[0])[i]);
            assertEquals(expected, itemCover.cardinality());
            i++;
        }
    }

    /**
     * Test the {@link CoverParallelCompute#pairCovers(Set)} method of the
     * {@link CoverParallelCompute} class.
     * This method tests the number of pairs returned by the
     * {@link CoverParallelCompute#pairCovers(Set)} method for different
     * item set sizes. If the number of items is odd, it checks that one of the
     * pairs contains a sparse bit set with all bits set.
     */
    @Test
    public void testPairCovers() {
        // Create an instance of CoverParallelCompute
        CoverParallelCompute coverParallelCompute = new CoverParallelCompute(dataset);

        // Items to test the cover compute function for
        String[] items = { "milk", "eggs", "butter", "cheese", "adult" };

        for (int i = 1; i <= items.length; i++) {
            Set<String> itemsInSet = new HashSet<>();
            for (int j = 0; j < i; j++) {
                itemsInSet.add(items[j]);
            }

            Set<SparseBitSet> covers = new HashSet<>();
            for (String item : items) {
                covers.add(itemsMap.get(item));
            }

            // Test the number of pairs
            Set<SparseBitSet[]> coverPairs = coverParallelCompute.pairCovers(covers);
            int expected = covers.size() / 2 + 1;
            assertEquals(expected, coverPairs.size());

            // If there is an odd number of covers, verify that one of the pairs contains a
            // sparse bit set with all bits set
            if (covers.size() % 2 == 1) {
                SparseBitSet SparseBitSetAllOnes = SetUtil.createSparseBitSetAllOnes(dataset.getNbTransactions());
                boolean hasAllOnes = false;
                for (SparseBitSet[] pair : coverPairs) {
                    if (pair[0].equals(SparseBitSetAllOnes) || pair[1].equals(SparseBitSetAllOnes)) {
                        hasAllOnes = true;
                        break;
                    }
                }
                assertTrue(hasAllOnes);
            }
        }
    }
}
