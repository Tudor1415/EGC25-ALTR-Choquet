package sampling;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.gson.Gson;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;
import tools.utils.RuleUtil;
import tools.utils.SetUtil;

public class RandomSampler {

    private static final int DEFAULT_ANTECEDENT_SIZE = 5;
    private static final double MIN_ANTECEDENT_FRACTION = 0.25;

    // Dataset and sampling-related variables
    private @Getter Dataset dataset;
    private @Getter String[] measureNames;
    private @Getter double smoothCounts;
    private @Getter @Setter int maxSizeX;
    private @Getter @Setter int maxSizeZ;

    // Random instance for this sampler
    private RandomUtil random = new RandomUtil();

    /**
     * Constructs a RandomSampler with the provided dataset, max sizes for
     * antecedent and consequent sets.
     *
     * @param dataset      The dataset used for sampling.
     * @param maxSizeX     The maximum size for the antecedent (X).
     * @param maxSizeZ     The maximum size for the consequent (Z).
     * @param nbSamples    The number of samples to generate.
     * @param measureNames Array of measure names.
     * @param smoothCounts The smoothing count for alternatives.
     */
    public RandomSampler(Dataset dataset, int maxSizeX, int maxSizeZ, String[] measureNames, double smoothCounts) {
        this.dataset = dataset;
        this.maxSizeX = maxSizeX;
        this.maxSizeZ = maxSizeZ;
        this.measureNames = measureNames;
        this.smoothCounts = smoothCounts;
    }

    /**
     * Samples a set of valid decision rules.
     *
     * @param numberOfRules     The number of rules to sample.
     * @param consequentItems   Available consequent items.
     * @param antecedentItems   Available antecedent items.
     * @param maxAntecedentSize Maximum size for the antecedent.
     * @return A set of randomly sampled valid decision rules.
     */
    public Set<DecisionRule> sample(int numberOfRules, Set<String> consequentItems,
            Set<String> antecedentItems, int maxAntecedentSize) {

        Map<IAlternative, DecisionRule> rulesSample = new HashMap<>();

        // Sample rules until the desired number of rules is reached
        while (rulesSample.keySet().size() < numberOfRules) {
            DecisionRule rule = new DecisionRule(new HashSet<>(), "", dataset, maxSizeX, maxSizeZ, smoothCounts,
                    measureNames);

            // Choose a random consequent item
            rule.setY(chooseRandomValue(rule, consequentItems, "consequent"));

            // Sample antecedent items
            sampleAntecedents(rule, antecedentItems, maxAntecedentSize);

            // Add the sampled rule to the set
            rulesSample.put(rule.getAlternative(),RuleUtil.deepCopy(rule));
        }

        return new HashSet<DecisionRule>(rulesSample.values());
    }

    /**
     * Samples antecedent values for the given rule.
     *
     * @param rule              The rule to add antecedent items to.
     * @param antecedentItems   Available antecedent items.
     * @param maxAntecedentSize Maximum size for the antecedent.
     */
    private void sampleAntecedents(DecisionRule rule, Set<String> antecedentItems, int maxAntecedentSize) {
        Set<String> availableAntecedents = SetUtil.copySet(antecedentItems);
        int antecedentSize = random
                .nextInt(max(min((int) (maxAntecedentSize * MIN_ANTECEDENT_FRACTION), DEFAULT_ANTECEDENT_SIZE),
                        DEFAULT_ANTECEDENT_SIZE))
                + 1;

        for (int i = 0; i < antecedentSize; i++) {
            String antecedentValue = chooseRandomValue(rule, availableAntecedents, "antecedent");

            if (antecedentValue.isEmpty()) {
                break;
            }

            rule.addToX(antecedentValue);
            availableAntecedents.remove(antecedentValue);
        }
    }

    /**
     * Chooses a random valid value for a rule, either for the antecedent or the
     * consequent.
     *
     * @param rule   The current rule.
     * @param values The set of possible values.
     * @param type   Whether choosing for "antecedent" or "consequent".
     * @return A randomly chosen valid value.
     */
    private String chooseRandomValue(DecisionRule rule, Set<String> values, String type) {
        if (isInitialValueFor(type, rule)) {
            return random.selectRandomElement(new ArrayList<>(values));
        }

        List<String> validValues = computeValidValues(rule, values, type);

        if ("antecedent".equals(type) && !rule.getItemsInX().isEmpty()) {
            validValues.add(""); // Empty antecedent option
        }

        return random.selectRandomElement(validValues);
    }

    private boolean isInitialValueFor(String type, DecisionRule rule) {
        return ("antecedent".equals(type) && rule.getItemsInX().isEmpty())
                || ("consequent".equals(type) && rule.getY().isEmpty());
    }

    /**
     * Computes the valid values that can be added to the rule.
     *
     * @param rule   The rule to check.
     * @param values The set of possible values.
     * @param type   "antecedent" or "consequent".
     * @return A list of valid values.
     */
    private List<String> computeValidValues(DecisionRule rule, Set<String> values, String type) {
        List<String> validValues = new ArrayList<>();
        DecisionRule ruleCopy = RuleUtil.deepCopy(rule);

        for (String value : values) {
            RuleUtil.addItemToRule(ruleCopy, value, type);

            if (RuleUtil.isValid(ruleCopy)) {
                validValues.add(value);
            }

            RuleUtil.removeItemFromRule(ruleCopy, value, type, rule.getY());
        }

        return validValues;
    }

    /**
     * Samples rules in parallel and saves them to a JSON file.
     *
     * @param numberOfRules   The number of rules to sample.
     * @param consequentItems Available consequent items.
     * @param antecedentItems Available antecedent items.
     * @param saveFilePath    File path to save the JSON.
     * @return A set of sampled rules.
     */
    public Set<DecisionRule> sampleToFile(int numberOfRules, Set<String> consequentItems,
            Set<String> antecedentItems, String saveFilePath) {

        Set<DecisionRule> sampledRules = sampleInParallel(numberOfRules, consequentItems, antecedentItems);

        saveRulesToFile(sampledRules, saveFilePath);

        return sampledRules;
    }

    /**
     * Samples rules in parallel based on available processors.
     *
     * @param numberOfRules   Number of rules to sample.
     * @param consequentItems Set of available consequent items.
     * @param antecedentItems Set of available antecedent items.
     * @return A set of sampled rules.
     */
    private Set<DecisionRule> sampleInParallel(int numberOfRules, Set<String> consequentItems,
            Set<String> antecedentItems) {
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int rulesPerTask = numberOfRules / availableProcessors;
        int maxAntecedentSize = calculateMaxAntecedentSize();

        return IntStream.range(0, availableProcessors)
                .parallel()
                .mapToObj(i -> sample(rulesPerTask, consequentItems, antecedentItems, maxAntecedentSize))
                .flatMap(Set::stream)
                .collect(Collectors.toSet());
    }

    /**
     * Calculates the maximum antecedent size based on the dataset's transactions.
     *
     * @return Maximum size for the antecedent.
     */
    private int calculateMaxAntecedentSize() {
        return (int) Math.round(Arrays.stream(dataset.getTransactions())
                .mapToDouble(row -> Arrays.stream(row).mapToInt(String::length).average().orElse(0))
                .average().orElse(0));
    }

    /**
     * Saves a set of rules to a file in JSON format.
     *
     * @param rules        The set of rules to save.
     * @param saveFilePath Path to save the JSON file.
     */
    private void saveRulesToFile(Set<DecisionRule> rules, String saveFilePath) {
        Gson gson = new Gson();
        String json = gson.toJson(rules);

        try (FileWriter writer = new FileWriter(saveFilePath)) {
            writer.write(json);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
