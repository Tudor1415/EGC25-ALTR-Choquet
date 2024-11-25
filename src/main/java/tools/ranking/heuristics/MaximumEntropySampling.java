package tools.ranking.heuristics;

import java.util.Map;
import java.util.Set;
import java.util.List;
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
import experiments.configs.MiningConfig;
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
    private Set<IAlternative[]> selectedPairs = new HashSet<>();
    private List<Ranking<IAlternative>> rankings = new ArrayList<>();
    private static final Logger logger = LoggerFactory.getLogger(MiningConfig.class.getSimpleName());

    public MaximumEntropySampling(MiningConfig config) {
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

        // Initialize sample
        initializeSample(config);
    }

    private int[] normalizeDominationVector(int[] dominationVector) {
        int[] negationVector = new int[dominationVector.length];
        for (int i = 0; i < dominationVector.length; i++) {
            negationVector[i] = -dominationVector[i];
        }
        // Choose the lexicographically larger vector as the canonical form
        for (int i = 0; i < dominationVector.length; i++) {
            if (dominationVector[i] > negationVector[i]) {
                return dominationVector;
            } else if (dominationVector[i] < negationVector[i]) {
                return negationVector;
            }
        }
        return dominationVector; // They are equal
    }

    private double calculateNormalizedEntropy(Map<int[], Integer> dominationCounts) {
        // Create a new map for normalized domination vectors
        Map<int[], Integer> normalizedCounts = new HashMap<>();

        // Aggregate counts for normalized domination vectors
        for (Map.Entry<int[], Integer> entry : dominationCounts.entrySet()) {
            int[] normalizedVector = normalizeDominationVector(entry.getKey());
            normalizedCounts.put(
                    normalizedVector,
                    normalizedCounts.getOrDefault(normalizedVector, 0) + entry.getValue());
        }

        // Compute entropy for the normalized counts
        double total = normalizedCounts.values().stream().mapToInt(Integer::intValue).sum();
        return -normalizedCounts.values().stream()
                .mapToDouble(count -> {
                    double prob = count / total;
                    return prob * Math.log(prob);
                })
                .sum();
    }

    private void initializeSample(MiningConfig config) {
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

    private int[] randomSample(int size, int sampleSize) {
        return random.kFolds(1, size, sampleSize)[0];
    }

    @Override
    public List<Ranking<IAlternative>> provideRankings(LearnStep step) {
        // Retrieve the state of the approximation function at the current iteration
        scoreFunction = step.getCurrentScoreFunction();
    
        // Ensure randomSampleSize is valid
        int sampleLength = sample.length;
        int actualSampleSize = Math.min(sampleLength, randomSampleSize);
    
        // Get random indices from the sample
        int[] randomSampleIndices = randomSample(sampleLength, actualSampleSize);
    
        // Initialize current domination counts
        Map<int[], Integer> dominationCounts = new HashMap<>();
        for (IAlternative[] pair : selectedPairs) {
            int[] dominationVector = computeDominationVector(pair[0], pair[1]);
            dominationCounts.put(dominationVector, dominationCounts.getOrDefault(dominationVector, 0) + 1);
        }
    
        // Variables to track the best pair
        double maxEntropy = Double.NEGATIVE_INFINITY;
        int bestAIndex = -1;
        int bestBIndex = -1;
    
        // Iterate over all pairs in the random sample
        for (int i = 0; i < actualSampleSize; i++) {
            for (int j = i + 1; j < actualSampleSize; j++) {
                int iIndex = randomSampleIndices[i];
                int jIndex = randomSampleIndices[j];
    
                IAlternative alt1 = sample[iIndex].getAlternative();
                IAlternative alt2 = sample[jIndex].getAlternative();
    
                // Skip if the pair has already been selected
                if (selectedPairs.contains(new IAlternative[] { alt1, alt2 })) {
                    continue;
                }
    
                // Compute domination vector and updated counts
                int[] dominationVector = computeDominationVector(alt1, alt2);
                dominationCounts.put(dominationVector, dominationCounts.getOrDefault(dominationVector, 0) + 1);
    
                // Calculate the entropy with the updated counts
                double entropy = calculateNormalizedEntropy(dominationCounts);
    
                // Update the best pair if this entropy is higher
                if (entropy > maxEntropy) {
                    maxEntropy = entropy;
                    bestAIndex = iIndex;
                    bestBIndex = jIndex;
                }
    
                // Revert the domination count for the current pair
                dominationCounts.put(dominationVector, dominationCounts.get(dominationVector) - 1);
            }
        }
    
        // If no suitable pair found, select the first two alternatives
        if (bestAIndex == -1 || bestBIndex == -1) {
            bestAIndex = 0;
            bestBIndex = 1;
        }
    
        // Prepare the selected alternatives
        IAlternative normA = new Alternative(
                normalizer.normalize(sample[bestAIndex].getAlternative().getVector(), normalizationMethod, false));
        IAlternative normB = new Alternative(
                normalizer.normalize(sample[bestBIndex].getAlternative().getVector(), normalizationMethod, false));
    
        IAlternative[] alternativePair = new IAlternative[] { normA, normB };
        List<DecisionRule> rulePair = new ArrayList<>();
        rulePair.add(sample[bestAIndex]);
        rulePair.add(sample[bestBIndex]);
    
        selectedPairs.add(alternativePair);
    
        // Compute the ranking for the selected pair using the oracle
        if (noise == 0) {
            rankings.add(RankingUtil.computeRankingWithOracle(oracle, rulePair, alternativePair));
        } else {
            rankings.add(RankingUtil.computeNoisyRankingWithOracle(oracle, rulePair, noise));
        }
    
        // Return all the computed rankings
        return rankings;
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
}
