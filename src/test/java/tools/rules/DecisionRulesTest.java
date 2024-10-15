package tools.rules;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import tools.data.Dataset;
import tools.utils.TestUtils;

public class DecisionRulesTest {

    @Test
    public void testGetFreq() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("4");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with sample X, Y, and dataset
        Set<String> x = new HashSet<>(Arrays.asList("1", "2"));
        String y = "4";
        DecisionRule rule = new DecisionRule(x, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Test the frequency calculation for X
        assertEquals(2, rule.getFreqX());

        // Test the frequency calculation for Y
        assertEquals(1, rule.getFreqY());

        // Test the frequency calculation for Z (union of X and Y)
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testSetX() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("3", "2"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Set a new X
        Set<String> newX = new HashSet<>(Arrays.asList("5", "2"));
        rule.setX(newX);

        // Test the frequency calculation for the updated X
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testAddToX() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("2"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Test the frequency calculation for X
        assertEquals(2, rule.getFreqX());

        // Add a new item to X
        rule.addToX("5");

        // Test the frequency calculation for the updated X
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testSetXMemoized() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("3", "2"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Set a new X
        Set<String> newX = new HashSet<>(Arrays.asList("5", "2"));
        rule.setX(newX);

        // Reset X to initial value (memoized)
        rule.setX(initialX);

        // Test the frequency calculation for the updated X
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testAddToXMemoized() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("3", "2"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Set a new X
        Set<String> newX = new HashSet<>(Arrays.asList("3"));
        rule.setX(newX);

        // Reset X to initial value bi adding "2" (memoized)
        rule.addToX("2");

        // Test the frequency calculation for the updated X
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testRemoveFromX() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("2", "3", "4"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Test the frequency calculation for X
        assertEquals(1, rule.getFreqX());

        // Remove an item from X
        rule.removeFromX("2");

        // Test the frequency calculation for the updated X
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testSetY() throws IOException {
        // Sample transactional dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("1");
        classItemValues.add("2");
        Dataset dataset = new Dataset("dataset_test_file.dat", "src/test/resources/", classItemValues);

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("3"));
        String y = "1";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Set a new Y
        rule.setY("2");

        // Test the frequency calculation for the updated Y
        assertEquals(1, rule.getFreqX());
        assertEquals(1, rule.getFreqZ());
    }

    @Test
    public void testComputeIrisBug() throws IOException {
        // Retrieving the iris dataset
        Set<String> classItemValues = new HashSet<>(Arrays.asList("13", "14", "15"));
        Dataset dataset = new Dataset("iris.dat", "src/test/resources/", classItemValues);
        String[][] irisTransactions = dataset.getTransactionalDataset();

        // Create Rule with initial X, Y, and dataset
        Set<String> initialX = new HashSet<>(Arrays.asList("4"));
        String y = "13";
        DecisionRule rule = new DecisionRule(initialX, y, dataset, 10, 10, 0.01d, new String[] { "confidence" });

        // Perform assertions on the computed cover
        int expectedX = TestUtils.countTransactionsWithItems(irisTransactions,
                rule.getItemsInX().toArray(new String[0]));
        assertEquals(expectedX, rule.getFreqX());
        int expectedZ = TestUtils.countTransactionsWithItems(irisTransactions,
                rule.getItemsInZ().toArray(new String[0]));
        assertEquals(expectedZ, rule.getFreqZ());

    }

}
