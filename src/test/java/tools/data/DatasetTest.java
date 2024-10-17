package tools.data;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import com.zaxxer.sparsebits.SparseBitSet;

import tools.utils.TestUtils;

public class DatasetTest {

    @Test
    public void testGetTransactionalDataset() {
        try {
            Set<String> classItemValues = new HashSet<>();
            classItemValues.add("1");

            Dataset Dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

            String[][] dataset = Dataset.getTransactionalDataset();

            assertNotNull("Dataset should not be null", dataset);
            assertEquals("Dataset size should match", 3, dataset.length);

            // Verify transaction values
            assertArrayEquals(new String[] { "1", "2", "3", "4" }, dataset[0]);
            assertArrayEquals(new String[] { "5", "2", "1" }, dataset[1]);
            assertArrayEquals(new String[] { "1", "9" }, dataset[2]);

        } catch (IOException e) {
            fail("IOException not expected during test");
        }
    }

    @Test
    public void testGetTransactionalDatasetStringItems() {
        try {
            Set<String> classItemValues = new HashSet<>(Arrays.asList("young", "adult", "senior"));
            Dataset Dataset = new Dataset("groceries_test_file.dat", "src/test/resources/", classItemValues);

            String[][] dataset = Dataset.getTransactionalDataset();

            assertNotNull("Dataset should not be null", dataset);
            assertEquals("Dataset size should match", 10, dataset.length);

            // Verify transaction values
            assertArrayEquals(new String[] { "milk", "bread", "eggs", "butter", "cheese", "young" }, dataset[0]);
            assertArrayEquals(new String[] { "bread", "eggs", "butter", "cheese", "yogurt", "adult" }, dataset[1]);
            assertArrayEquals(new String[] { "milk", "bread", "butter", "cheese", "yogurt", "adult" }, dataset[2]);
            assertArrayEquals(new String[] { "milk", "eggs", "butter", "cheese", "yogurt", "senior" }, dataset[3]);
            assertArrayEquals(new String[] { "milk", "bread", "eggs", "cheese", "yogurt", "senior" }, dataset[4]);
            assertArrayEquals(new String[] { "bread", "eggs", "butter", "cheese", "yogurt", "young" }, dataset[5]);
            assertArrayEquals(new String[] { "milk", "bread", "eggs", "butter", "adult" }, dataset[6]);
            assertArrayEquals(new String[] { "milk", "bread", "eggs", "cheese", "adult" }, dataset[7]);
            assertArrayEquals(new String[] { "bread", "eggs", "butter", "cheese", "senior" }, dataset[8]);
            assertArrayEquals(new String[] { "milk", "bread", "butter", "cheese", "young" }, dataset[9]);
        } catch (IOException e) {
            fail("IOException not expected during test");
        }
    }

    @Test
    public void testClassItems() {
        try {
            Set<String> classItemValues = new HashSet<>();
            classItemValues.add("13");
            classItemValues.add("14");
            classItemValues.add("15");

            Dataset Dataset = new Dataset("iris.dat", "src/test/resources/", classItemValues);

            Set<String> classItems = Dataset.getConsequentItemsSet();

            assertNotNull("The set of class items should not be null", classItems);
            assertEquals("Number of class items should match", 3, classItems.size());

            assertSetContainsValue("13", classItems);
            assertSetContainsValue("14", classItems);
            assertSetContainsValue("15", classItems);

        } catch (IOException e) {
            fail("IOException not expected during test");
        }
    }

    @Test
    public void testAntecedentItems() {
        try {
            Set<String> classItemValues = new HashSet<>();
            classItemValues.add("13");
            classItemValues.add("14");
            classItemValues.add("15");

            Dataset Dataset = new Dataset("iris.dat", "src/test/resources/", classItemValues);

            Set<String> antecedentItems = Dataset.getAntecedentItemsSet();

            assertNotNull("The set of antecedent items should not be null", antecedentItems);
            assertEquals("Number of antecedent items should match", 12, antecedentItems.size());

            assertSetContainsValue("1", antecedentItems);
            assertSetContainsValue("2", antecedentItems);
            assertSetContainsValue("3", antecedentItems);
            assertSetContainsValue("4", antecedentItems);
            assertSetContainsValue("5", antecedentItems);
            assertSetContainsValue("6", antecedentItems);
            assertSetContainsValue("7", antecedentItems);
            assertSetContainsValue("8", antecedentItems);
            assertSetContainsValue("9", antecedentItems);
            assertSetContainsValue("10", antecedentItems);
            assertSetContainsValue("11", antecedentItems);
            assertSetContainsValue("12", antecedentItems);

        } catch (IOException e) {
            fail("IOException not expected during test");
        }
    }

    /**
     * Test the cover computation for the iris dataset.
     * This method retrieves the iris dataset using the {@link Dataset} class,
     * retrieves the covers for each item in the dataset using the
     * {@link Dataset#getItemsMap()} method,
     * and checks that the covers have the correct size using the
     * {@link TestUtils#countTransactionsWithItems(String[][], String...)}
     * method.
     *
     * @throws IOException if there is an error reading the dataset file
     */
    @Test
    public void testCoverComputationIris() throws IOException {
        // Retrieving the iris dataset
        Set<String> classItemValues = new HashSet<>(Arrays.asList("13", "14", "15"));
        Dataset irisDataset = new Dataset("iris.dat", "src/test/resources/", classItemValues);
        String[][] irisTransactions = irisDataset.getTransactionalDataset();

        for (String itemValue : irisDataset.getItemsMap().keySet()) {
            SparseBitSet cover = irisDataset.getItemsMap().get(itemValue);
            assertNotNull(cover);

            // Perform assertions on the computed cover
            int expected = TestUtils.countTransactionsWithItems(irisTransactions, itemValue);
            assertEquals(expected, cover.cardinality());
        }
    }

    @Test
    public void testGetItemsFromTransactions() {
        try {
            Set<String> classItemValues = new HashSet<>();
            classItemValues.add("1");

            Dataset Dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

            Map<String, SparseBitSet> items = Dataset.getItemsMap();

            assertNotNull("Items should not be null", items);
            assertEquals("Number of items should match", 6, items.size());

            // Verify item values and occurrences
            int[] occurrencesForItemValue1 = { 0, 1, 2 };
            int[] occurrencesForItemValue2 = { 0, 1 };
            int[] occurrencesForItemValue3 = { 0 };
            int[] occurrencesForItemValue4 = { 0 };
            int[] occurrencesForItemValue5 = { 1 };
            int[] occurrencesForItemValue6 = { 2 };

            // Example assertions
            assertItem("1", createSparseBitSet(occurrencesForItemValue1), items);
            assertItem("2", createSparseBitSet(occurrencesForItemValue2), items);
            assertItem("3", createSparseBitSet(occurrencesForItemValue3), items);
            assertItem("4", createSparseBitSet(occurrencesForItemValue4), items);
            assertItem("5", createSparseBitSet(occurrencesForItemValue5), items);
            assertItem("9", createSparseBitSet(occurrencesForItemValue6), items);

        } catch (IOException e) {
            fail("IOException not expected during test");
        }
    }

    private static SparseBitSet createSparseBitSet(int[] values) {
        SparseBitSet bitSet = new SparseBitSet();
        for (int value : values) {
            bitSet.set(value);
        }
        return bitSet;
    }

    private void assertItem(String valueToTest, SparseBitSet expectedOccurrences, Map<String, SparseBitSet> items) {
        assertEquals("Occurrences should match", expectedOccurrences, items.get(valueToTest));
    }

    private void assertSetContainsValue(String value, Set<String> itemsValues) {
        assertTrue("Item with value " + value + " not found in the array",
                itemsValues.contains(value));
    }

    @Test
    public void testFindEquivalenceClasses() throws IOException {
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");

        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        dataset.findEquivalenceClasses();

        UnionFind uf = dataset.getEquivalenceClasses();

        assertTrue("1 should be connected with 2", uf.find("1").equals(uf.find("2")));
        assertTrue("1 should be connected with 3", uf.find("1").equals(uf.find("3")));
        assertTrue("1 should be connected with 4", uf.find("1").equals(uf.find("4")));
        assertTrue("1 should be connected with 5", uf.find("1").equals(uf.find("5")));
        assertTrue("1 should be connected with 9", uf.find("1").equals(uf.find("9")));
        assertTrue("2 should be connected with 5", uf.find("2").equals(uf.find("5")));
        assertTrue("2 should be connected with 3", uf.find("2").equals(uf.find("3")));
        assertTrue("2 should be connected with 4", uf.find("2").equals(uf.find("4")));
    }

    private Set<String> getClassItems(String datasetName) {
        switch (datasetName) {
            case "adult":
                return new HashSet<>(Arrays.asList("145", "146"));
            case "bank":
                return new HashSet<>(Arrays.asList("89", "90"));
            case "connect":
                return new HashSet<>(Arrays.asList("127", "128"));
            case "credit":
                return new HashSet<>(Arrays.asList("111", "112"));
            case "dota":
                return new HashSet<>(Arrays.asList("346", "347"));
            case "toms":
                return new HashSet<>(Arrays.asList("911", "912"));
            case "mushroom":
                return new HashSet<>(Arrays.asList("116", "117"));
            default:
                return null;
        }
    }

    @Test
    public void datasetEquivalenceClasses() throws IOException {
        String datasetName = "mushroom";

        Dataset dataset = new Dataset(datasetName+".dat", "src/test/resources/", getClassItems(datasetName));

        dataset.findEquivalenceClasses();

        UnionFind uf = dataset.getEquivalenceClasses();

        System.out.println("Dataset " + datasetName + " has " + uf.countClasses() + " equivalence class(es)." );
    }
}
