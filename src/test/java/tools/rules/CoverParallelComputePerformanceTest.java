package tools.rules;

import com.zaxxer.sparsebits.SparseBitSet;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import tools.data.Dataset;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CoverParallelComputePerformanceTest {

    private static Dataset largeDataset;
    private static Set<String> itemsInSet;

    @BeforeAll
    public static void setUp() throws IOException {
        // Create a large dataset for performance testing
        // For demonstration, we simulate a large dataset
        // In practice, you should load an actual large dataset
        int numberOfTransactions = 100_000_000;  // Adjust size as needed for testing
        int numberOfItems = 1000;

        // Generate synthetic dataset
        largeDataset = generateLargeDataset(numberOfTransactions, numberOfItems);

        // Select a subset of items to compute the cover
        itemsInSet = new HashSet<>();
        Random rand = new Random();
        for (int i = 0; i < 10; i++) {  // Adjust the number of items as needed
            itemsInSet.add("Item_" + rand.nextInt(numberOfItems));
        }
    }

    /**
     * Test to measure and compare the performance of GPU vs CPU computation.
     */
    @Test
    public void testComputePerformance() {
        // Initialize CoverParallelCompute for CPU
        CoverParallelCompute cpuCompute = new CoverParallelCompute(largeDataset);
        cpuCompute.onGPU = false;  // Force CPU computation

        // Initialize CoverParallelCompute for GPU
        CoverParallelCompute gpuCompute = new CoverParallelCompute(largeDataset);
        gpuCompute.onGPU = true;  // Ensure GPU computation

        // Warm up the GPU (optional but recommended)
        gpuCompute.computeGPU(itemsInSet);

        // Measure CPU computation time
        long startTimeCpu = System.nanoTime();
        SparseBitSet cpuResult = cpuCompute.compute(itemsInSet);
        long endTimeCpu = System.nanoTime();
        long durationCpu = TimeUnit.NANOSECONDS.toMillis(endTimeCpu - startTimeCpu);

        // Measure GPU computation time
        long startTimeGpu = System.nanoTime();
        int[] gpuResult = gpuCompute.computeGPU(itemsInSet);
        long endTimeGpu = System.nanoTime();
        long durationGpu = TimeUnit.NANOSECONDS.toMillis(endTimeGpu - startTimeGpu);

        // Output the results
        System.out.println("CPU computation time: " + durationCpu + " ms");
        System.out.println("GPU computation time: " + durationGpu + " ms");

        // Verify that both results are the same
        assertEquals(cpuResult, gpuResult, "The CPU and GPU results should be identical.");

        // Optionally, assert that GPU is faster (depends on your hardware and dataset size)
        // assertTrue(durationGpu < durationCpu, "GPU computation should be faster than CPU computation.");
    }

    /**
     * Generates a synthetic large dataset for testing purposes.
     *
     * @param numberOfTransactions Number of transactions.
     * @param numberOfItems        Number of unique items.
     * @return A Dataset object containing the synthetic data.
     */
    private static Dataset generateLargeDataset(int numberOfTransactions, int numberOfItems) {
        // Create synthetic transactional data
        String[][] transactions = new String[numberOfTransactions][];
        Random rand = new Random();

        for (int i = 0; i < numberOfTransactions; i++) {
            int itemsInTransaction = rand.nextInt(20) + 1;  // Each transaction has between 1 and 20 items
            Set<String> transactionItems = new HashSet<>();
            for (int j = 0; j < itemsInTransaction; j++) {
                transactionItems.add("Item_" + rand.nextInt(numberOfItems));
            }
            transactions[i] = transactionItems.toArray(new String[0]);
        }

        // Generate class item values (for demonstration, we can just use some placeholders)
        Set<String> classItemValues = new HashSet<>(Arrays.asList("Class1", "Class2", "Class3"));

        // Create the Dataset object
        Dataset dataset = new Dataset(transactions, classItemValues);
        dataset.setNbTransactions(numberOfTransactions);

        return dataset;
    }
}
