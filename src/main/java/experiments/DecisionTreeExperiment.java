package experiments;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.zaxxer.sparsebits.SparseBitSet;

import sampling.DecisionTreeSampling.AlgorithmType;
import sampling.DecisionTreeSampling.DecisionTreeSampler;
import sampling.DecisionTreeSampling.TreeNode;
import tools.data.Dataset;
import tools.rules.DecisionRule;

public class DecisionTreeExperiment {

    private static final String dataDirectory = "data/folds/";

    public static void main(String[] args) {
        try {
            runExperiment();
        } catch (IOException e) {
            System.err.println("Error during the experiment: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void runExperiment() throws IOException {
        List<String> datasetNames = Arrays.asList("bank");
        List<Double> allGains = new ArrayList<>();

        for (String datasetName : datasetNames) {
            List<Dataset> testDatasets = readDatasetsFromFold(datasetName, "Test");
            for (Dataset testDataset : testDatasets) {
                int classIndex = 0; // Ensure this index is correctly set as per your dataset schema

                // Convert the array of arrays to a list of arrays
                List<String[]> transactionList = Arrays.asList(testDataset.getTransactions());

                DecisionTreeSampler sampler = new DecisionTreeSampler(
                        transactionList,
                        testDataset.getAntecedentItemsSet(),
                        classIndex,
                        AlgorithmType.ID3,
                        10,
                        testDataset.getItemsMap(),
                        2);

                TreeNode root = sampler.buildTree();
                List<DecisionRule> rules = sampler.extractRules(root, testDataset, 1.0,
                        new String[] { "measure1", "measure2" });
                double maxGain = rules.stream()
                        .mapToDouble(rule -> calculateInformationGain(rule, testDataset))
                        .max()
                        .orElse(0);
                allGains.add(maxGain);
            }
        }

        double averageGain = allGains.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        System.out.println("Average Information Gain across all datasets and folds: " + averageGain);
    }

    private static List<Dataset> readDatasetsFromFold(String datasetName, String trainOrTest) throws IOException {
        String folderPath = dataDirectory + datasetName + "/" + trainOrTest;
        File folder = new File(folderPath);

        List<Dataset> datasets = new ArrayList<>();

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();

            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        Dataset dataset = new Dataset(file.getPath(), folderPath, getClassItems(datasetName));
                        datasets.add(dataset);
                    }
                }
            } else {
                System.out.println("The folder is empty or cannot be read.");
            }
        } else {
            System.out.println("The specified path is not a valid directory.");
        }

        return datasets;
    }

    private static Set<String> getClassItems(String datasetName) {
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

    /**
     * Method to calculate information gain based on the decision rule's impacts on
     * dataset segmentation.
     * 
     * @param rule    The decision rule for which to calculate information gain.
     * @param dataset The dataset containing the transactions and item covers.
     * @return The calculated information gain.
     */
    public static double calculateInformationGain(DecisionRule rule, Dataset dataset) {
        SparseBitSet allTransactionIndices = new SparseBitSet();
        allTransactionIndices.set(0, dataset.getNbTransactions()); // Assuming all transactions are indexed from 0 to
                                                                   // n-1

        // Get the itemset cover for the antecedent from the rule
        SparseBitSet antecedentCover = rule.getCoverX();

        // Calculate total entropy before the split
        double totalEntropy = calculateEntropy(allTransactionIndices, dataset);

        // Calculate entropy after the split into antecedent and non-antecedent
        // transactions
        SparseBitSet includedIndices = antecedentCover; // Transactions that include the antecedent
        SparseBitSet excludedIndices = allTransactionIndices.clone();
        excludedIndices.andNot(antecedentCover); // Transactions that exclude the antecedent

        double includedEntropy = calculateEntropy(includedIndices, dataset);
        double excludedEntropy = calculateEntropy(excludedIndices, dataset);

        int totalInstances = allTransactionIndices.cardinality();
        double includedWeight = includedIndices.cardinality() / (double) totalInstances;
        double excludedWeight = excludedIndices.cardinality() / (double) totalInstances;

        double weightedEntropy = includedWeight * includedEntropy + excludedWeight * excludedEntropy;

        return totalEntropy - weightedEntropy;
    }

    /**
     * Calculates the entropy of a set of transactions within the dataset.
     * 
     * @param transactionIndices The indices of transactions to consider.
     * @param dataset            The dataset from which to pull class labels and
     *                           other data.
     * @return The entropy value calculated.
     */
    private static double calculateEntropy(SparseBitSet transactionIndices, Dataset dataset) {
        Map<String, Integer> classCounts = countClasses(transactionIndices, dataset);
        int totalInstances = transactionIndices.cardinality();
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = (double) count / totalInstances;
                entropy -= probability * Math.log(probability) / Math.log(2);
            }
        }
        return entropy;
    }

    /**
     * Counts the occurrences of each class in the provided set of transaction
     * indices
     * by using the covers of class items and intersecting them with the specified
     * transaction indices.
     *
     * @param transactionIndices A SparseBitSet indicating the indices of
     *                           transactions to consider.
     * @param dataset            The dataset containing all transactions, including
     *                           class labels and their covers.
     * @return A map from class labels to their count within the specified
     *         transaction indices.
     */
    public static Map<String, Integer> countClasses(SparseBitSet transactionIndices, Dataset dataset) {
        Map<String, Integer> classCounts = new HashMap<>();
        Set<String> classItems = dataset.getConsequentItemsSet();

        for (String classItem : classItems) {
            SparseBitSet classItemCover = dataset.getItemsMap().get(classItem);
            if (classItemCover != null) {
                SparseBitSet relevantTransactions = classItemCover.clone();
                relevantTransactions.and(transactionIndices);
                int count = relevantTransactions.cardinality();
                classCounts.put(classItem, count);
            }
        }

        return classCounts;
    }

}