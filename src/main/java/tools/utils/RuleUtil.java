package tools.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.google.gson.annotations.SerializedName;

import lombok.Getter;
import tools.data.Dataset;
import tools.rules.DecisionRule;
import tools.rules.IRule;

public class RuleUtil {
    /**
     * Saves a copy with critical info only.
     *
     * @param originalRule The rule to copy.
     * @return A saved copy of the original rule.
     */
    public static DecisionRule simpleCopy(DecisionRule originalRule) {
        // Copy the contents of the sets to new sets
        Set<String> newX = new HashSet<>(originalRule.getItemsInX());
        String newY = originalRule.getY();

        // Create a new Rule with the copied sets
        DecisionRule copiedRule = DecisionRule.builder()
                .itemsInX(newX)
                .Y(newY)
                .alternative(originalRule.getAlternative() != null ? originalRule.getAlternative().deepCopy() : null)
                .freqX(originalRule.getFreqX())
                .freqY(originalRule.getFreqY())
                .freqZ(originalRule.getFreqZ())
                .smoothCounts(1e-6d)
                .build();

        return copiedRule;
    }

    /**
     * Saves a copy with all the information.
     *
     * @param originalRule The rule to copy.
     * @return A saved copy of the original rule.
     */
    public static DecisionRule deepCopy(DecisionRule originalRule) {
        return originalRule.toBuilder()
                .dataset(originalRule.getDataset())
                .itemsMap(new HashMap<>(originalRule.getItemsMap()))
                .freqX(originalRule.getFreqX())
                .freqY(originalRule.getFreqY())
                .freqZ(originalRule.getFreqZ())
                .smoothCounts(originalRule.getSmoothCounts())
                .measureNames(originalRule.getMeasureNames() != null ? originalRule.getMeasureNames().clone() : null)
                .alternative(originalRule.getAlternative() != null ? originalRule.getAlternative().deepCopy() : null)
                .coverX(originalRule.getCoverX() != null ? originalRule.getCoverX().clone() : null)
                .coverY(originalRule.getCoverY() != null ? originalRule.getCoverY().clone() : null)
                .coverZ(originalRule.getCoverZ() != null ? originalRule.getCoverZ().clone() : null)
                .memoizedCoverX(
                        originalRule.getMemoizedCoverX() != null ? new HashMap<>(originalRule.getMemoizedCoverX())
                                : null)
                .memoizedCoverZ(
                        originalRule.getMemoizedCoverZ() != null ? new HashMap<>(originalRule.getMemoizedCoverZ())
                                : null)
                .Y(new String(originalRule.getY()))
                .itemsInX(SetUtil.copySet(originalRule.getItemsInX()))
                .itemsInZ(SetUtil.copySet(originalRule.getItemsInZ()))
                .build();
    }

    /**
     * Checks if a rule is valid.
     *
     * @param rule The rule to be checked for validity.
     * @return {@code true} if the rule is valid, {@code false} otherwise.
     */
    public static boolean isValid(IRule rule) {
        // Check if frequencies are not zero and X and Y are not empty
        if ((rule.getFreqZ() > 0 && rule.getFreqX() > 0 && rule.getFreqY() > 0) && !rule.getItemsInX().isEmpty()
                && rule.getY() != null) {
            return true;
        }

        return false;
    }

    static class Rule {
        @SerializedName("Antecedent")
        private @Getter String antecedent;

        @SerializedName("Consequent")
        private @Getter String consequent;

        @SerializedName("Frequency")
        private @Getter int frequency;

        @SerializedName("Confidence")
        private @Getter double confidence;
    }

    public static Set<String> convertStringToSet(String input) {
        // Remove brackets "[" and "]"
        String[] elements = input.substring(1, input.length() - 1).split(", ");

        // Create a set to store the elements
        Set<String> resultSet = new HashSet<>();

        // Add each element to the set
        for (String element : elements) {
            resultSet.add(element);
        }

        return resultSet;
    }

    public static String convertStringClassToClass(String input) {
        // Remove brackets "[" and "]"
        String[] elements = input.substring(1, input.length() - 1).split(", ");

        // Create a List to store the elements
        List<String> resultList = new ArrayList<>();

        // Add each element to the List
        for (String element : elements) {
            resultList.add(element);
        }

        return resultList.get(0);
    }

    private static int[] randomSample(int size, int sampleSize) {
        RandomUtil random = RandomUtil.getInstance();
        return random.kFolds(1, size, sampleSize)[0];
    }

    /**
     * This function takes the path to the file, the dataset, and the measure names,
     * then generates a list of DecisionRules.
     * 
     * @param filePath     Path to the CSV file.
     * @param dataset      The dataset object.
     * @param measureNames The measure names used for rule generation.
     * @return A list of DecisionRules.
     * @throws IOException If an error occurs while reading the file.
     */
    public static DecisionRule[] extractRulesFromCSV(String filePath, Dataset dataset, String[] measureNames)
            throws IOException {

        List<DecisionRule> decisionRules = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;

            // Skip the first header line
            reader.readLine();

            // Read and process each line from the CSV file
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");

                // Parse the antecedent, consequent, and frequencies
                String antecedentRaw = parts[0].replace("{", "").replace("}", "");
                String[] antecedentItems = antecedentRaw.split(";");

                String consequentRaw = parts[1].replace("{", "").replace("}", "");
                String[] consequentItems = consequentRaw.split(";");

                // Create the sets of antecedent and consequent items
                Set<String> itemsInX = new HashSet<>();
                for (String item : antecedentItems) {
                    itemsInX.add(item.trim());
                }

                // Assume consequent contains one item for simplicity
                String consequent = consequentItems[0].trim();

                // Create and configure the decision rule
                DecisionRule rule = new DecisionRule(itemsInX, consequent, dataset, 100, 100, 1e-6, measureNames);

                // Add the decision rule to the list
                decisionRules.add(rule);
            }
        }

        return decisionRules.toArray(new DecisionRule[0]);
    }
    /**
     * Adds an item to the antecedent or consequent of a rule.
     * 
     * @param J         The rule to which the item will be added.
     * @param itemValue The value of the item to be added.
     * @param type      Indicates if the item is a consequent or antecedent item.
     */
    public static void addItemToRule(IRule J, String itemValue, String type) {
        if (type.equals("antecedent")) {
            J.addToX(itemValue); // Adding the item to the antecedent of the rule
        } else if (type.equals("consequent")) {
            J.setY(itemValue); // Adding the item to the consequent of the rule
        }
    }

    /**
     * Removes an item from the antecedent or consequent of a rule.
     * 
     * @param J         The rule from which the item will be removed.
     * @param itemValue The value of the item to be removed.
     * @param type      Indicates if the item is a consequent or antecedent item.
     * @param lastClass The last class of the rule in the case of the consequent.
     */
    public static void removeItemFromRule(IRule J, String itemValue, String type, String lastClass) {
        if (type.equals("antecedent")) {
            J.removeFromX(itemValue); // Removing the item from the antecedent of the rule
        } else if (type.equals("consequent")) {
            J.setY(lastClass); // Restoring the consequent of the rule
        }
    }
}
