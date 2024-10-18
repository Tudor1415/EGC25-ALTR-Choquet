package tools.rules;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.Test;
import org.junit.jupiter.api.BeforeAll;

public class CoverParallelComputeCUDATest {
    public static int[] generateRandomIntArray(int size) {
        Random random = new Random();
        int[] array = new int[size];
    
        for (int i = 0; i < size; i++) {
            array[i] = random.nextInt(2);
        }
    
        return array;
    }    

    private static final int[] TEST_SIZES = { 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000 };

    @Test
    public void testBitwiseAndPerformance() {
        for (int size : TEST_SIZES) {
            System.out.println("Testing with array size: " + size);

            // Generate test data
            int[] cover1 = generateRandomIntArray(size);
            int[] cover2 = generateRandomIntArray(size);
            int[] resultCpu = new int[size];
            int[] resultGpu = new int[size];

            // Measure CPU execution time
            // long cpuStartTime = System.nanoTime();
            // CpuBitwiseAnd.computeAndOperation(cover1, cover2, resultCpu);
            // long cpuEndTime = System.nanoTime();
            // double cpuDurationMs = (cpuEndTime - cpuStartTime) / 1_000_000.0;

            // Measure GPU execution time
            long gpuStartTime = System.nanoTime();
            CoverParallelComputeCUDA.runCudaKernel(cover1, cover2, resultGpu, size);
            long gpuEndTime = System.nanoTime();
            double gpuDurationMs = (gpuEndTime - gpuStartTime) / 1_000_000.0;

            // Verify correctness of results
            // assertArrayEquals(resultCpu, resultGpu, "The GPU and CPU results should be identical");

            // Output execution times and speedup
            // System.out.printf("CPU Time: %.2f ms%n", cpuDurationMs);
            System.out.printf("GPU Time: %.2f ms%n", gpuDurationMs);
            // System.out.printf("Speedup: %.2f%n", (cpuDurationMs / gpuDurationMs));

            // Ensure that GPU is faster or reasonably competitive
            // assertTrue(cpuDurationMs > gpuDurationMs || gpuDurationMs < cpuDurationMs * 2,
                    // "GPU should perform at least as fast as the CPU.");
        }
    }
}
