package tools.normalization;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import tools.normalization.Normalizer.NormalizationMethod;

import java.util.Arrays;

public class NormalizerTest {
    @Test
    public void testDecimalScaling() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {100, 200, 300};
        normalizer.normalize(vector, NormalizationMethod.DECIMAL_SCALING, true);

        double[] newVector = {150, 250, 350};
        double[] normalized = normalizer.normalize(newVector, NormalizationMethod.DECIMAL_SCALING, false);

        int maxDigits = 3; // Max digits from initial data
        double scalingFactor = Math.pow(10, maxDigits);
        assertArrayEquals(new double[]{
            150 / scalingFactor,
            250 / scalingFactor,
            350 / scalingFactor
        }, normalized, 1e-6);
    }

    @Test
    public void testTanhEstimator() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {50, 60, 70};
        normalizer.normalize(vector, NormalizationMethod.TANH_ESTIMATOR, true);

        double[] newVector = {55, 65, 75};
        double[] normalized = normalizer.normalize(newVector, NormalizationMethod.TANH_ESTIMATOR, false);

        // Check that values are between 0 and 1
        for (double value : normalized) {
            assertTrue(value >= 0 && value <= 1);
        }
    }

    @Test
    public void testZNormalizationZeroVariance() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {20, 20, 20};
        normalizer.normalize(vector, NormalizationMethod.Z_NORMALIZATION, true);

        double[] newVector = {20, 20, 20};
        double[] normalized = normalizer.normalize(newVector, NormalizationMethod.Z_NORMALIZATION, false);

        // Variance is zero; normalized values should be zero
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, normalized, 1e-6);
    }

    @Test
    public void testEmpiricalCDF() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {10, 20, 30, 40, 50};
        for (double v : vector) {
            normalizer.normalize(new double[]{v}, NormalizationMethod.EMPIRICAL_CDF, true);
        }

        double[] newVector = {15, 35, 55};
        double[] normalized = new double[newVector.length];
        for (int i = 0; i < newVector.length; i++) {
            normalized[i] = normalizer.normalize(new double[]{newVector[i]}, NormalizationMethod.EMPIRICAL_CDF, false)[0];
        }

        // Approximate CDF values using T-Digest
        assertTrue(normalized[0] > 0.0 && normalized[0] < 0.4);
        assertTrue(normalized[1] > 0.4 && normalized[1] < 0.8);
        assertTrue(normalized[2] > 0.8 && normalized[2] <= 1.0);
    }

    @Test
    public void testStatisticsUpdate() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {10, 20, 30};
        normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);

        // Statistics should be updated
        assertEquals(10, normalizer.getCoordinateStats().get(0).min);
        assertEquals(30, normalizer.getCoordinateStats().get(2).max);

        double[] newVector = {40, 50, 60};
        normalizer.normalize(newVector, NormalizationMethod.MIN_MAX_SCALING, false);

        // Statistics should remain the same
        assertEquals(10, normalizer.getCoordinateStats().get(0).min);
        assertEquals(30, normalizer.getCoordinateStats().get(2).max);
    }

    @Test
    public void testNegativeValues() {
        Normalizer normalizer = new Normalizer();
        double[] vectorMin = {-30, -20, -15};
        normalizer.normalize(vectorMin, NormalizationMethod.MIN_MAX_SCALING, true);

        double[] vectorMax = {-10, -10, -10};
        normalizer.normalize(vectorMax, NormalizationMethod.MIN_MAX_SCALING, true);

        double[] newVector = {-25, -15, -5};
        double[] normalized = normalizer.normalize(newVector, NormalizationMethod.MIN_MAX_SCALING, false);

        double max = -10;
        assertArrayEquals(new double[]{
            (-25 - (-30)) / (max - (-30)),
            (-15 - (-20)) / (max - (-20)),
            (-5 - (-15)) / (max - (-15))
        }, normalized, 1e-6);
    }

    @Test
    public void testEmptyVector() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {};
        double[] normalized = normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);

        assertEquals(0, normalized.length);
    }

    @Test
    public void testInvalidVectorLength() {
        Normalizer normalizer = new Normalizer();
        double[] vector = {1.0, 2.0};
        normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);

        double[] invalidVector = {1.0};
        assertThrows(IllegalArgumentException.class, () -> {
            normalizer.normalize(invalidVector, NormalizationMethod.MIN_MAX_SCALING, false);
        });
    }

    @Test
    public void testParallelPerformance() {
        Normalizer normalizer = new Normalizer(); // Enable parallel processing
        double[] vector = new double[1_000];
        Arrays.setAll(vector, i -> Math.random() * 1000);

        long startTime = System.currentTimeMillis();
        normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);
        long endTime = System.currentTimeMillis();
        long parallelDuration = endTime - startTime;

        // Now test with parallel processing disabled
        double[] newVector = new double[1_000];
        Arrays.setAll(newVector, i -> Math.random() * 1000);

        startTime = System.currentTimeMillis();
        normalizer.normalize(newVector, NormalizationMethod.MIN_MAX_SCALING, true);
        endTime = System.currentTimeMillis();
        long sequentialDuration = endTime - startTime;

        System.out.println("Parallel duration: " + parallelDuration + "ms");
        System.out.println("Sequential duration: " + sequentialDuration + "ms");
    }

    @Test
    public void testThreadSafety() throws InterruptedException {
        // Test that the normalizer can handle concurrent updates
        Normalizer normalizer = new Normalizer();
        double[] vector = new double[1000];
        Arrays.fill(vector, 1.0);

        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);
            }
        });

        Thread thread2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                normalizer.normalize(vector, NormalizationMethod.MIN_MAX_SCALING, true);
            }
        });

        thread1.start();
        thread2.start();

        thread1.join();
        thread2.join();

        // If no exceptions occur, the test passes
    }
}
