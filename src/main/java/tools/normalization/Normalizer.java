package tools.normalization;

import java.util.*;
import java.util.stream.IntStream;

import com.tdunning.math.stats.TDigest;

/**
 * This class takes in a vector of doubles and, given a normalization method,
 * returns a new normalized vector.
 * The different methods are:
 * - Min max scaling: requires the historical minimum and maximum for each
 * coordinate
 * - Mean normalization: requires the minimum, maximum, and mean historical
 * value for each coordinate
 * - Maximum Absolute Scaling: requires the maximum historical value for each
 * coordinate
 * - Median normalization: requires the historical median value for each
 * coordinate
 * - Decimal scaling: requires the historical maximum number of digits for each
 * coordinate
 * - Tanh Estimator: requires the historical empirical mean and variance for
 * each coordinate
 * - Z normalization: requires the historical mean and variance for each
 * coordinate
 * - Empirical CDF: uses T-Digest to estimate quantiles for each coordinate
 * 
 * Historical values are numerically derived from the sample of alternatives
 * seen.
 * This class has a 'save' parameter for the normalize function.
 * If 'save' is true, then all the required historical values are updated.
 */
public class Normalizer {

    public enum NormalizationMethod {
        NO_NORMALIZATION,
        MIN_MAX_SCALING,
        MEAN_NORMALIZATION,
        MAX_ABSOLUTE_SCALING,
        MEDIAN_NORMALIZATION,
        DECIMAL_SCALING,
        TANH_ESTIMATOR,
        Z_NORMALIZATION,
        EMPIRICAL_CDF
    }

    class CoordinateStats {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        private double sum = 0.0;
        private double sumSq = 0.0;
        private int count = 0;

        private double mean = Double.NaN;
        private double variance = Double.NaN;
        private double stddev = Double.NaN;
        private boolean statsDirty = true;

        private TDigest tDigest;

        private double maxAbs = 0.0;

        private int maxDigits = 0;

        public CoordinateStats() {
            tDigest = TDigest.createDigest(100);
        }

        public void update(double value) {
            count++;
            sum += value;
            sumSq += value * value;
            min = Math.min(min, value);
            max = Math.max(max, value);
            maxAbs = Math.max(maxAbs, Math.abs(value));
            int digits = countDigits(value);
            maxDigits = Math.max(maxDigits, digits);
            tDigest.add(value);
            statsDirty = true;
        }

        private void recomputeStats() {
            if (statsDirty && count > 0) {
                mean = sum / count;
                variance = (sumSq - (sum * sum) / count) / count;
                variance = Math.max(variance, 0.0);
                stddev = Math.sqrt(variance);
                statsDirty = false;
            }
        }

        public double getMean() {
            recomputeStats();
            return mean;
        }

        public double getVariance() {
            recomputeStats();
            return variance;
        }

        public double getStddev() {
            recomputeStats();
            return stddev;
        }

        public double getMedian() {
            return tDigest.quantile(0.5);
        }

        public double getCDF(double value) {
            if (tDigest.size() == 0) {
                return 0.0;
            }
            return tDigest.cdf(value);
        }

    }

    private List<CoordinateStats> coordinateStats;
    private int dimensions;

    public Normalizer() {
        coordinateStats = new ArrayList<>();
    }

    public double[] normalize(double[] vector, NormalizationMethod method, boolean save) {
        if (dimensions == 0) {
            dimensions = vector.length;
        } else if (vector.length != dimensions) {
            throw new IllegalArgumentException("Vector length must be consistent.");
        }

        while (coordinateStats.size() < dimensions) {
            coordinateStats.add(new CoordinateStats());
        }

        double[] result = new double[dimensions];

        IntStream.range(0, dimensions).forEach(i -> {
            double value = vector[i];
            CoordinateStats stats = coordinateStats.get(i);

            if (save) {
                stats.update(value);
            }

            switch (method) {
                case MIN_MAX_SCALING:
                    double min = stats.min;
                    double max = stats.max;
                    result[i] = max != min ? (value - min) / (max - min) : 0.0;
                    break;
                case MEAN_NORMALIZATION:
                    double mean = stats.getMean();
                    min = stats.min;
                    max = stats.max;
                    result[i] = max != min ? (value - mean) / (max - min) : 0.0;
                    break;
                case MAX_ABSOLUTE_SCALING:
                    double maxAbs = stats.maxAbs;
                    result[i] = maxAbs != 0.0 ? value / maxAbs : 0.0;
                    break;
                case MEDIAN_NORMALIZATION:
                    double median = stats.getMedian();
                    result[i] = median != 0.0 ? value / median : 0.0;
                    break;
                case DECIMAL_SCALING:
                    int j = stats.maxDigits;
                    double scalingFactor = Math.pow(10, j);
                    result[i] = scalingFactor != 0.0 ? value / scalingFactor : 0.0;
                    break;
                case TANH_ESTIMATOR:
                    mean = stats.getMean();
                    double stddev = stats.getStddev();
                    result[i] = stddev != 0.0 ? 0.5 * (Math.tanh(0.01 * ((value - mean) / stddev))) + 0.5 : 0.5;
                    break;
                case Z_NORMALIZATION:
                    mean = stats.getMean();
                    stddev = stats.getStddev();
                    result[i] = stddev != 0.0 ? (value - mean) / stddev : 0.0;
                    break;
                case EMPIRICAL_CDF:
                    if (stats.count > 0) {
                        result[i] = stats.getCDF(value);
                    } else {
                        result[i] = 0.0;
                    }
                    break;

                default:
                    result[i] = value; // No normalization applied
                    break;
            }
        });

        return result;
    }

    private int countDigits(double value) {
        value = Math.abs(value);
        return value == 0 ? 1 : (int) Math.floor(Math.log10(value)) + 1;
    }

    public List<CoordinateStats> getCoordinateStats() {
        return coordinateStats;
    }
}
