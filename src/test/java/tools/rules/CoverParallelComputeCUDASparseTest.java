package tools.rules;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

public class CoverParallelComputeCUDASparseTest {
    private static final int[] TEST_SIZES = { 1_000, 10_000, 100_000, 1_000_000 };

    // Generate random sparse index arrays with a given number of non-zero indices
    private int[] generateRandomIndices(int arraySize, int numNonZeros) {
        Random random = new Random();
        Set<Integer> indicesSet = new HashSet<>();

        while (indicesSet.size() < numNonZeros) {
            int index = random.nextInt(arraySize);
            indicesSet.add(index);
        }

        return indicesSet.stream().mapToInt(Integer::intValue).sorted().toArray();
    }

    // Calculate intersection on the CPU for correctness check
    private int[] computeIntersectionCPU(int[] indices1, int[] indices2) {
        return IntStream.of(indices1)
                .filter(value -> Arrays.binarySearch(indices2, value) >= 0)
                .toArray();
    }

    @Test
    public void testSparseAndPerformance() {
        for (int size : TEST_SIZES) {
            System.out.println("Testing with array size: " + size);

            // Generate random sparse index arrays
            int numNonZeros1 = size / 100; // 1% density
            int numNonZeros2 = size / 100;

            int[] indices1 = generateRandomIndices(size, numNonZeros1);
            int[] indices2 = generateRandomIndices(size, numNonZeros2);

            // Sort the arrays (just in case, though generation already sorts them)
            Arrays.sort(indices1);
            Arrays.sort(indices2);

            // Compute expected intersection on the CPU
            int[] expectedIntersection = computeIntersectionCPU(indices1, indices2);

            // Allocate space for GPU result
            int[] resultIndicesGPU = new int[expectedIntersection.length];

            // Measure GPU execution time
            long gpuStartTime = System.nanoTime();
            CoverParallelComputeCUDA.runCudaSparseAnd(indices1, indices2, resultIndicesGPU);
            long gpuEndTime = System.nanoTime();
            double gpuDurationMs = (gpuEndTime - gpuStartTime) / 1_000_000.0;

            // Trim result array to the actual size of the intersection (if smaller)
            int[] actualIntersection = Arrays.copyOf(resultIndicesGPU, expectedIntersection.length);

            // Verify correctness of results
            assertArrayEquals(expectedIntersection, actualIntersection, "The GPU and CPU results should be identical");

            // Output execution times
            System.out.printf("GPU Time: %.2f ms%n", gpuDurationMs);

            // Ensure GPU is reasonably competitive
            assertTrue(gpuDurationMs < 1000, "GPU execution time should be under 1 second for this test size.");
        }
    }
}
