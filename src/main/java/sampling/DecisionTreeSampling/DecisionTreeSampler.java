package sampling.DecisionTreeSampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.zaxxer.sparsebits.SparseBitSet;

import sampling.Sampler;
import tools.data.Dataset;
import tools.rules.DecisionRule;

public class DecisionTreeSampler implements Sampler {

    private Dataset dataset;
    private Set<String> items;
    private AlgorithmType algorithmType;
    private int maxDepth;
    private Map<String, SparseBitSet> itemsMap;
    private int maxItemsetSize;

    public DecisionTreeSampler(Dataset dataset, Set<String> items, AlgorithmType algorithmType,
            int maxDepth, Map<String, SparseBitSet> itemsMap, int maxItemsetSize) {
        this.dataset = dataset;
        this.items = items;
        this.algorithmType = algorithmType;
        this.maxDepth = maxDepth;
        this.itemsMap = itemsMap;
        this.maxItemsetSize = maxItemsetSize;
    }

    public TreeNode buildTree() {
        SparseBitSet transactionIndices = new SparseBitSet();
        transactionIndices.set(0, dataset.getNbTransactions());
        return buildTreeRecursive(transactionIndices, new HashSet<>(), 0);
    }

    private TreeNode buildTreeRecursive(SparseBitSet transactionIndices, Set<Set<String>> usedItemsets, int depth) {
        // Leaf Node
        if (isPure(transactionIndices) || depth >= maxDepth) {
            return new TreeNode(null, getMostCommonClass(transactionIndices));
        }

        List<Set<String>> candidateItemsets = generateCandidateItemsets(usedItemsets);
        Set<String> bestItemset = null;
        double bestMetric = Double.MIN_VALUE;

        for (Set<String> itemset : candidateItemsets) {
            if (usedItemsets.contains(itemset))
                continue;

            double metric = calculateMetric(transactionIndices, itemset);

            if (metric > bestMetric) {
                bestMetric = metric;
                bestItemset = itemset;
            }
        }

        // Leaf Node
        if (bestItemset == null) {
            return new TreeNode(null, getMostCommonClass(transactionIndices));
        }

        TreeNode node = new TreeNode(bestItemset, getMostCommonClass(transactionIndices));
        usedItemsets.add(bestItemset);

        SparseBitSet itemsetCover = getItemsetCover(bestItemset);
        SparseBitSet includedIndices = itemsetCover.clone();
        includedIndices.and(transactionIndices);
        SparseBitSet excludedIndices = transactionIndices.clone();
        excludedIndices.andNot(itemsetCover);

        node.includedChild = buildTreeRecursive(includedIndices, new HashSet<>(usedItemsets), depth + 1);
        node.excludedChild = buildTreeRecursive(excludedIndices, new HashSet<>(usedItemsets), depth + 1);

        return node;
    }

    private double calculateMetric(SparseBitSet transactionIndices, Set<String> itemset) {
        double metric = 0.0;
        switch (algorithmType) {
            case ID3:
                metric = calculateInformationGain(transactionIndices, itemset);
                break;
            case C45:
                metric = calculateGainRatio(transactionIndices, itemset);
                break;
            case CART:
                metric = calculateGiniInformationGain(transactionIndices, itemset);
                break;
        }
        return metric;
    }

    // Generate candidate itemsets up to maxItemsetSize
    private List<Set<String>> generateCandidateItemsets(Set<Set<String>> usedItemsets) {
        List<Set<String>> candidateItemsets = new ArrayList<>();
        for (int size = 1; size <= maxItemsetSize; size++) {
            candidateItemsets.addAll(generateItemsetsOfSize(size, usedItemsets));
        }
        return candidateItemsets;
    }

    private List<Set<String>> generateItemsetsOfSize(int size, Set<Set<String>> usedItemsets) {
        List<Set<String>> itemsets = new ArrayList<>();
        Set<String> availableItems = new HashSet<>(items);
        for (Set<String> usedItemset : usedItemsets) {
            availableItems.removeAll(usedItemset);
        }
        itemsets = generateCombinations(new ArrayList<>(availableItems), size);
        return itemsets;
    }

    // Generate combinations of items of a given size
    private List<Set<String>> generateCombinations(List<String> itemsList, int size) {
        List<Set<String>> combinations = new ArrayList<>();
        combine(itemsList, combinations, new HashSet<>(), 0, size);
        return combinations;
    }

    private void combine(List<String> itemsList, List<Set<String>> combinations, Set<String> current, int index,
            int size) {
        if (current.size() == size) {
            combinations.add(new HashSet<>(current));
            return;
        }
        for (int i = index; i < itemsList.size(); i++) {
            current.add(itemsList.get(i));
            combine(itemsList, combinations, current, i + 1, size);
            current.remove(itemsList.get(i));
        }
    }

    // Compute the cover for an itemset
    private SparseBitSet getItemsetCover(Set<String> itemset) {
        Iterator<String> iterator = itemset.iterator();
        SparseBitSet cover = itemsMap.get(iterator.next()).clone();
        while (iterator.hasNext()) {
            cover.and(itemsMap.get(iterator.next()));
        }
        return cover;
    }

    // Check if the subset of transactions is pure (all have the same class label)
    private boolean isPure(SparseBitSet transactionIndices) {
        String firstClass = null;
        int index = transactionIndices.nextSetBit(0);
        while (index >= 0) {
            String[] transaction = dataset.getTransactions()[index];
            String classLabel = transaction[transaction.length - 1]; // Use last item as class label
            if (firstClass == null) {
                firstClass = classLabel;
            } else if (!firstClass.equals(classLabel)) {
                return false;
            }
            index = transactionIndices.nextSetBit(index + 1);
        }
        return true;
    }

    // Get the most common class label in the subset of transactions
    private String getMostCommonClass(SparseBitSet transactionIndices) {
        Map<String, Integer> classCounts = new HashMap<>();
        int index = transactionIndices.nextSetBit(0);
        while (index >= 0) {
            String[] transaction = dataset.getTransactions()[index];
            String classLabel = transaction[transaction.length - 1]; // Use last item as class label
            classCounts.put(classLabel, classCounts.getOrDefault(classLabel, 0) + 1);
            index = transactionIndices.nextSetBit(index + 1);
        }
        return Collections.max(classCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Calculate entropy
    private double calculateEntropy(SparseBitSet transactionIndices) {
        Map<String, Integer> classCounts = new HashMap<>();
        int totalInstances = transactionIndices.cardinality();
        int index = transactionIndices.nextSetBit(0);
        while (index >= 0) {
            String[] transaction = dataset.getTransactions()[index];
            String classLabel = transaction[transaction.length - 1]; // Use last item as class label
            classCounts.put(classLabel, classCounts.getOrDefault(classLabel, 0) + 1);
            index = transactionIndices.nextSetBit(index + 1);
        }
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            double probability = (double) count / totalInstances;
            entropy -= probability * (Math.log(probability) / Math.log(2));
        }
        return entropy;
    }

    // Calculate Gini impurity
    private double calculateGiniImpurity(SparseBitSet transactionIndices) {
        Map<String, Integer> classCounts = new HashMap<>();
        int totalInstances = transactionIndices.cardinality();
        int index = transactionIndices.nextSetBit(0);
        while (index >= 0) {
            String[] transaction = dataset.getTransactions()[index];
            String classLabel = transaction[transaction.length - 1]; // Use last item as class label
            classCounts.put(classLabel, classCounts.getOrDefault(classLabel, 0) + 1);
            index = transactionIndices.nextSetBit(index + 1);
        }
        double impurity = 1.0;
        for (int count : classCounts.values()) {
            double probability = (double) count / totalInstances;
            impurity -= probability * probability;
        }
        return impurity;
    }

    // Calculate Information Gain for ID3
    private double calculateInformationGain(SparseBitSet transactionIndices, Set<String> itemset) {
        double totalEntropy = calculateEntropy(transactionIndices);

        // Compute the cover for the itemset
        SparseBitSet itemsetCover = getItemsetCover(itemset);

        // Split transactions based on the inclusion of the itemset
        SparseBitSet includedIndices = itemsetCover.clone();
        includedIndices.and(transactionIndices);

        SparseBitSet excludedIndices = transactionIndices.clone();
        excludedIndices.andNot(itemsetCover);

        int totalInstances = transactionIndices.cardinality();

        double includedEntropy = calculateEntropy(includedIndices);
        double excludedEntropy = calculateEntropy(excludedIndices);

        double includedWeight = (double) includedIndices.cardinality() / totalInstances;
        double excludedWeight = (double) excludedIndices.cardinality() / totalInstances;

        double weightedEntropy = includedWeight * includedEntropy + excludedWeight * excludedEntropy;

        return totalEntropy - weightedEntropy;
    }

    // Calculate Gain Ratio for C4.5
    private double calculateGainRatio(SparseBitSet transactionIndices, Set<String> itemset) {
        double infoGain = calculateInformationGain(transactionIndices, itemset);

        // Compute the cover for the itemset
        SparseBitSet itemsetCover = getItemsetCover(itemset);

        // Split info
        int totalInstances = transactionIndices.cardinality();

        SparseBitSet includedIndices = itemsetCover.clone();
        includedIndices.and(transactionIndices);

        SparseBitSet excludedIndices = transactionIndices.clone();
        excludedIndices.andNot(itemsetCover);

        double includedWeight = (double) includedIndices.cardinality() / totalInstances;
        double excludedWeight = (double) excludedIndices.cardinality() / totalInstances;

        double splitInfo = 0.0;
        if (includedWeight > 0) {
            splitInfo -= includedWeight * (Math.log(includedWeight) / Math.log(2));
        }
        if (excludedWeight > 0) {
            splitInfo -= excludedWeight * (Math.log(excludedWeight) / Math.log(2));
        }

        return (splitInfo == 0) ? 0 : infoGain / splitInfo;
    }

    // Calculate Information Gain using Gini impurity for CART
    private double calculateGiniInformationGain(SparseBitSet transactionIndices, Set<String> itemset) {
        double totalGini = calculateGiniImpurity(transactionIndices);

        // Compute the cover for the itemset
        SparseBitSet itemsetCover = getItemsetCover(itemset);

        // Split transactions based on the inclusion of the itemset
        SparseBitSet includedIndices = itemsetCover.clone();
        includedIndices.and(transactionIndices);

        SparseBitSet excludedIndices = transactionIndices.clone();
        excludedIndices.andNot(itemsetCover);

        int totalInstances = transactionIndices.cardinality();

        double includedGini = calculateGiniImpurity(includedIndices);
        double excludedGini = calculateGiniImpurity(excludedIndices);

        double includedWeight = (double) includedIndices.cardinality() / totalInstances;
        double excludedWeight = (double) excludedIndices.cardinality() / totalInstances;

        double weightedGini = includedWeight * includedGini + excludedWeight * excludedGini;

        return totalGini - weightedGini;
    }

    // Method to extract rules from the decision tree
    public List<DecisionRule> extractRules(TreeNode node, Dataset dataset, double smoothCounts, String[] measureNames) {
        List<DecisionRule> rules = new ArrayList<>();
        extractRulesRecursive(node, new HashSet<>(), rules, dataset, smoothCounts, measureNames);
        return rules;
    }

    private void extractRulesRecursive(TreeNode node, Set<String> conditions,
            List<DecisionRule> rules, Dataset dataset, double smoothCounts, String[] measureNames) {
        if (node.isLeaf()) {
            // Create a new DecisionRule
            Set<String> antecedent = new HashSet<>(conditions);
            DecisionRule rule = new DecisionRule(antecedent, node.classLabel, dataset, 100, 100, smoothCounts,
                    measureNames);
            rules.add(rule);
            return;
        }

        // Include the itemset
        Set<String> includedConditions = new HashSet<>(conditions);
        includedConditions.addAll(node.itemset);
        extractRulesRecursive(node.includedChild, includedConditions, rules, dataset, smoothCounts, measureNames);

        // Exclude the itemset
        Set<String> excludedConditions = new HashSet<>(conditions);
        // Optionally note that the itemset is excluded
        extractRulesRecursive(node.excludedChild, excludedConditions, rules, dataset, smoothCounts, measureNames);
    }

    public List<DecisionRule> sample() {
        TreeNode root = this.buildTree();
        List<DecisionRule> rules = this.extractRules(root, dataset, 1.0, new String[] { "support", "confidence" });

        return rules;
    }

}
