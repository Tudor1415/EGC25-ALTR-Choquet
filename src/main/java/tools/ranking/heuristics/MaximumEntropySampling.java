package tools.ranking.heuristics;

import java.util.Map;
import java.util.Set;
import java.util.List;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.io.IOException;
import java.util.ArrayList;
import java.util.stream.Collectors;

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

        // Find a valid target vector with a non-empty list
        for (Map.Entry<DominationVector, List<DecisionRule[]>> entry : dominationMap.entrySet()) {
            if (!entry.getValue().isEmpty()) {
                targetVector = entry.getKey();
                selectedPair = entry.getValue().remove(0);
                break;
            }
        }

        // Handle case where no valid pair is found
        if (selectedPair == null) {
            throw new IllegalStateException("No valid decision rule pair found in domination map!");
        }

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

@Getter
@Setter
class DominationVector {
    private final int[] vector;

    public DominationVector(int[] vector) {
        this.vector = vector;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        DominationVector that = (DominationVector) o;
        return Arrays.equals(vector, that.vector);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(vector);
    }

    @Override
    public String toString() {
        return Arrays.toString(vector);
    }
}
