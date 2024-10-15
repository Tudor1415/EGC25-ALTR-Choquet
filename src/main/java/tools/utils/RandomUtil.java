package tools.utils;

import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Utility class for generating random values using the Java Random class.
 */

public class RandomUtil {

    private Random random = new Random();
    private static RandomUtil INSTANCE;

    /**
     * Returns the singleton instance of the RandomUtil class.
     *
     * @return The singleton instance of the RandomUtil class.
     */
    public static RandomUtil getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new RandomUtil();
        }
        return INSTANCE;
    }

    /**
     * Generate k array of equal length of integers, where each array represents the
     * indexes of the training data for a fold
     * 
     * @param k    number of folds
     * @param size number of data rows
     * @return the k folds
     */
    public int[][] kFolds(int k, int size) {
        List<Integer> idx = IntStream.range(0, size).boxed().collect(Collectors.toList());
        Collections.shuffle(idx, random);
        int foldSize = size / k;
        int[][] folds = new int[k][];
        for (int i = 0; i < k; i++) {
            folds[i] = new int[foldSize];
            for (int j = 0; j < foldSize; j++) {
                folds[i][j] = idx.get(i * foldSize + j);
            }
        }
        return folds;
    }

    /**
     * Generate k array of integers, where each array represents the indexes of the
     * training data for a fold
     * 
     * @param k        number of folds
     * @param size     number of data rows
     * @param foldSize size of each fold
     * @return the k folds
     */
    public int[][] kFolds(int k, int size, int foldSize) {
        List<Integer> idx = IntStream.range(0, size).boxed().collect(Collectors.toList());
        int[][] folds = new int[k][];
        for (int i = 0; i < k; i++) {
            Collections.shuffle(idx, random);
            folds[i] = new int[foldSize];
            for (int j = 0; j < foldSize; j++) {
                folds[i][j] = idx.get(j);
            }
        }
        return folds;
    }

    /**
     * k folds but returns for each int between [0, size-1] the corresponding folds
     * ex: [0,1,1,0] means that objects at index 0 and 3 belong to fold 0, objects
     * at index 1 and 2 to fold 1
     * 
     * @param k
     * @param size
     * @return
     */
    public int[] kFoldsIndex(int k, int size) {
        int[][] kFolds = kFolds(k, size);
        int[] foldsIndex = new int[size];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < kFolds[i].length; j++) {
                int idx = kFolds[i][j];
                foldsIndex[idx] = i;
            }
        }
        return foldsIndex;
    }

    /**
     * Generate an array of random doubles in [0,1] such that the sum of the array
     * is equal to 1
     * 
     * @param nbWeights size of the array
     * @return the array of random doubles
     */
    public double[] generateRandomWeights(int nbWeights) {
        assert nbWeights >= 1;
        double[] weights = new double[nbWeights];
        double sum = 0;
        for (int i = 0; i < nbWeights; i++) {
            weights[i] = -1 * Math.log(1.0 - random.nextDouble()) + 0.01;
            sum += weights[i];
        }
        for (int i = 0; i < nbWeights; i++) {
            weights[i] = weights[i] / sum;
        }
        return weights;
    }

    /**
     * Generates random Choquet capacities for a given number of criteria.
     *
     * @param nbCriteria The number of criteria for which capacities are generated.
     * @return A map containing random Choquet capacities represented by BitSets and
     *         their corresponding values.
     */
    public Map<BitSet, Double> generateRandomChoquetCapacities(int nbCriteria) {
        Map<BitSet, Double> capacities = new HashMap<>();
        int nbCapacity = (int) Math.pow(2, nbCriteria);

        // Generate random capacities for all non-empty subsets
        for (int i = 1; i < nbCapacity - 1; i++) {
            BitSet capacity = SetUtil.intToBitSet(i, nbCriteria);
            double capacityValue = random.nextDouble();
            capacities.put(capacity, capacityValue);
        }

        // Set capacities for the empty and full subsets
        capacities.put(SetUtil.intToBitSet(0, nbCriteria), 0d);
        capacities.put(SetUtil.intToBitSet(nbCapacity - 1, nbCriteria), 1d);

        // Ensure monotonicity by updating capacities
        for (BitSet a : capacities.keySet()) {
            for (BitSet b : capacities.keySet()) {
                if (SetUtil.isSubsetOf(a, b) && capacities.get(a) > capacities.get(b)) {
                    capacities.put(b, capacities.get(a));
                }
            }
        }

        return capacities;
    }

    /**
     * Returns the next pseudorandom, uniformly distributed double value between 0.0
     * and 1.0.
     *
     */
    public double nextDouble() {
        return random.nextDouble();
    }

    /**
     * Returns a pseudorandom, uniformly distributed int value between 0 (inclusive)
     * and the specified bound (exclusive).
     *
     * @param bound The upper bound (exclusive). Must be positive.
     * @return A pseudorandom, uniformly distributed int value between 0 (inclusive)
     *         and the specified bound (exclusive).
     * @throws IllegalArgumentException If bound is not positive.
     */
    public int nextInt(int bound) {
        return random.nextInt(bound);
    }

    /**
     * Sets the seed of this random number generator using a single long seed.
     *
     * @param seed The initial seed.
     */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    /**
     * A Bernoulli trial of parameter p
     * 
     * @param p The probability of a success i.e. returning a 1
     * @return The result of the Bernoulli trial of probability p
     */
    public boolean Bernoulli(double p) {
        return nextDouble() <= p;
    }

    /**
     * Generates an array of random vectors with specified dimensions.
     *
     * @param n          The number of vectors to generate.
     * @param vectorSize The size of each vector.
     * @return An array of random vectors with dimensions [n][vectorSize].
     */
    public double[][] generateRandomVectors(int n, int vectorSize) {
        double[][] randomVectors = new double[n][];
        for (int i = 0; i < n; i++) {
            randomVectors[i] = new double[vectorSize];
            for (int j = 0; j < vectorSize; j++) {
                randomVectors[i][j] = random.nextDouble();
            }
        }
        return randomVectors;
    }

    /**
     * Constructs the file path for a specific fold of a dataset.
     *
     * @param alternativesPath The base path of the dataset file.
     * @param type             The type of fold (e.g., "train", "test").
     * @param foldIdx          The index of the fold.
     * @return The constructed file path for the specified fold.
     */
    public static String getFoldPath(String alternativesPath, String type, int foldIdx) {
        return alternativesPath.substring(0, alternativesPath.length() - 5) + type + foldIdx + ".json";
    }

    /**
     * Shuffles the elements of a List using the specified randomization source.
     *
     * @param l The List to be shuffled.
     */
    public void shuffle(List<?> l) {
        Collections.shuffle(l, random);
    }

    /**
     * Selects a random element from a List.
     *
     * @param l   The List from which to select a random element.
     * @param <T> The type of elements in the List.
     * @return A randomly selected element from the List.
     */
    public <T> T selectRandomElement(List<T> l) {
        return l.get(nextInt(l.size()));
    }

    /**
     * Selects a random element from an array of integers up to a specified maximum
     * index.
     *
     * @param elt      The array of integers.
     * @param maxIndex The maximum index (exclusive).
     * @return A randomly selected element from the array.
     */
    public int selectRandomElement(int[] elt, int maxIndex) {
        return elt[nextInt(maxIndex)];
    }

    /**
     * Generates a random shuffle of integers from 0 to n-1.
     *
     * @param n the number of integers to shuffle
     * @return a shuffled array of integers
     */
    public static int[] randomShuffle(int n) {
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = i;
        }

        Random rand = new Random();
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

        return arr;
    }

    /**
     * Chooses a value randomly.
     *
     * @param values The set of possible values.
     * @return The randomly chosen value.
     */
    public static String chooseUniformRandom(Set<String> values) {
        int size = values.size();
        int item = ThreadLocalRandom.current().nextInt(size);
        int i = 0;
        for (String value : values) {
            if (i == item) {
                return value;
            }
            i++;
        }
        // This point is reached only if the size of the set is 0
        // In this case, return null
        return null;
    }
}