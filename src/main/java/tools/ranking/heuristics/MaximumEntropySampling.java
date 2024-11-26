package tools.ranking.heuristics;

import java.util.Map;
import java.util.Set;
import java.util.List;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.io.IOException;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.utils.RuleUtil;
import tools.ranking.Ranking;
import tools.train.LearnStep;
import tools.utils.RandomUtil;
import tools.utils.RankingUtil;
import tools.rules.DecisionRule;
import tools.alternatives.Alternative;
import tools.normalization.Normalizer;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.alternatives.IAlternative;
import experiments.configs.MaximumEntropySamplingConfig;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;

@Getter
@Setter
public class MaximumEntropySampling implements RankingsProvider {

    // Configurable parameters
    private double noise;
    private int randomSampleSize;
    private NormalizationMethod normalizationMethod;

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;
    private DecisionRule[] sample;

    // Internal components
    private Normalizer normalizer;
    private RandomUtil random;
    private ISinglevariateFunction scoreFunction;

    // State
    private Map<DominationVector, List<DecisionRule[]>> dominationMap = new HashMap<>();
    private Map<DominationVector, Integer> dominationCounts = new HashMap<>();
    private Set<IAlternative[]> selectedPairs = new HashSet<>();
    private Set<int[]> normalizedDominationVectors;
    private List<Ranking<IAlternative>> rankings = new ArrayList<>();
    private static final Logger logger = LoggerFactory.getLogger(MaximumEntropySampling.class.getSimpleName());

    public MaximumEntropySampling(MaximumEntropySamplingConfig config) {
        // Set configurable parameters
        this.noise = config.getNoise();
        this.randomSampleSize = config.getRandomSampleSize();
        this.normalizationMethod = config.getNormalizationMethod();

        // Set required components
        this.oracle = config.getOracle();
        this.dataset = config.getDataset();
        this.measureNames = config.getMeasureNames();

        // Initialize components
        this.normalizer = new Normalizer();
        this.random = RandomUtil.getInstance();

        // Initialize sample and domination map
        initializeSample(config);
        setupDominationMap();
    }

    private void setupDominationMap() {
        logger.info("Setting up domination map...");

        // Initialize dominationMap and dominationCounts
        dominationMap = new HashMap<>();
        dominationCounts = new HashMap<>();

        // Populate map with decision rule pairs
        for (int i = 0; i < sample.length; i++) {
            for (int j = i + 1; j < sample.length; j++) {
                IAlternative alt1 = sample[i].getAlternative();
                IAlternative alt2 = sample[j].getAlternative();

                // Normalize the domination vector
                DominationVector dominationVector = new DominationVector(
                        normalizeDominationVector(computeDominationVector(alt1, alt2)));

                // Initialize map entries if they don't exist
                dominationMap.computeIfAbsent(dominationVector, k -> new ArrayList<>());
                dominationCounts.putIfAbsent(dominationVector, 0);

                // Add the rule pair
                dominationMap.get(dominationVector).add(new DecisionRule[] { sample[i], sample[j] });
            }
        }

        logger.info("Domination map setup completed. Total vectors: {}", dominationMap.size());
    }

    private void initializeSample(MaximumEntropySamplingConfig config) {
        try {
            logger.info("Initializing sample from rules path: {}", config.getOutputPath());

            this.sample = RuleUtil.extractRulesFromCSV(config.getOutputPath(), dataset, measureNames);

            logger.info("Successfully loaded {} rules from {}", sample.length, config.getOutputPath());

            // Normalize the sample
            for (DecisionRule rule : sample) {
                normalizer.normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);
            }

            logger.info("Sample normalization completed for {} rules.", sample.length);

        } catch (IOException e) {
            logger.error("Failed to load rules from path: {}. Error: {}", config.getOutputPath(), e.getMessage(), e);
        }
    }

    @Override
    public List<Ranking<IAlternative>> provideRankings(LearnStep step) {
        // Retrieve the state of the approximation function at the current iteration
        scoreFunction = step.getCurrentScoreFunction();

        DominationVector targetVector = null;
        DecisionRule[] selectedPair = null;

        // Calculate entropy of domination counts
        double maxEntropy = Double.NEGATIVE_INFINITY;

        for (Map.Entry<DominationVector, List<DecisionRule[]>> entry : dominationMap.entrySet()) {
            List<DecisionRule[]> pairs = entry.getValue();

            // Skip empty lists
            if (pairs.isEmpty())
                continue;

            // Temporarily update the domination counts
            DominationVector vector = entry.getKey();
            dominationCounts.put(vector, dominationCounts.getOrDefault(vector, 0) + 1);

            // Calculate the entropy with the updated counts
            double entropy = calculateEntropy(dominationCounts);

            // Revert the domination counts to the original state
            dominationCounts.put(vector, dominationCounts.get(vector) - 1);

            // Select the vector with maximum entropy
            if (entropy > maxEntropy) {
                maxEntropy = entropy;
                targetVector = vector;
                selectedPair = pairs.get(0); // Don't remove yet; only selecting
            }
        }

        // Handle case where no valid pair is found
        if (selectedPair == null) {
            throw new IllegalStateException("No valid decision rule pair found in domination map!");
        }

        // Remove the selected pair from the map
        dominationMap.get(targetVector).remove(0);

        // Update domination counts
        dominationCounts.put(targetVector, dominationCounts.get(targetVector) + 1);

        // Normalize and prepare alternatives
        IAlternative normA = new Alternative(
                normalizer.normalize(selectedPair[0].getAlternative().getVector(), normalizationMethod, false));
        IAlternative normB = new Alternative(
                normalizer.normalize(selectedPair[1].getAlternative().getVector(), normalizationMethod, false));

        IAlternative[] alternativePair = new IAlternative[] { normA, normB };
        selectedPairs.add(alternativePair);

        // Compute the ranking for the selected pair using the oracle
        if (noise == 0) {
            rankings.add(RankingUtil.computeRankingWithOracle(oracle, Arrays.asList(selectedPair), alternativePair));
        } else {
            rankings.add(RankingUtil.computeNoisyRankingWithOracle(oracle, Arrays.asList(selectedPair), noise));
        }

        return rankings;
    }

    private double calculateEntropy(Map<DominationVector, Integer> counts) {
        double total = counts.values().stream().mapToDouble(Integer::doubleValue).sum();

        // If total is 0, entropy is undefined; handle gracefully
        if (total == 0)
            return 0;

        return counts.values().stream()
                .mapToDouble(count -> {
                    double probability = count / total;
                    return probability > 0 ? -probability * Math.log(probability) : 0;
                })
                .sum();
    }

    private int[] computeDominationVector(IAlternative alt1, IAlternative alt2) {
        int[] dominationVector = new int[alt1.getVector().length];
        for (int i = 0; i < alt1.getVector().length; i++) {
            if (alt1.getVector()[i] > alt2.getVector()[i]) {
                dominationVector[i] = 1;
            } else if (alt1.getVector()[i] < alt2.getVector()[i]) {
                dominationVector[i] = -1;
            } else {
                dominationVector[i] = 0;
            }
        }
        return dominationVector;
    }

    private int[] normalizeDominationVector(int[] dominationVector) {
        int[] negationVector = new int[dominationVector.length];
        for (int i = 0; i < dominationVector.length; i++) {
            negationVector[i] = -dominationVector[i];
        }
        for (int i = 0; i < dominationVector.length; i++) {
            if (dominationVector[i] > negationVector[i]) {
                return dominationVector;
            } else if (dominationVector[i] < negationVector[i]) {
                return negationVector;
            }
        }
        return dominationVector; // They are equal
    }
}
