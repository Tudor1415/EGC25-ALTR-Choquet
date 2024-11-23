package tools.ranking.heuristics;

import static java.lang.Math.abs;

import java.util.Set;
import java.util.List;
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
public class UncertaintyMining implements RankingsProvider {

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

    public UncertaintyMining(MiningConfig config) {
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

    private void initializeSample(MiningConfig config) {
        try {
            logger.info("Initializing sample from rules path: {}", config.getRulesPath());

            this.sample = RuleUtil.extractRulesFromCSV(config.getOutputPath(), dataset, measureNames);

            logger.info("Successfully loaded {} rules from {}", sample.length, config.getRulesPath());

            // Normalize the sample
            for (DecisionRule rule : sample) {
                normalizer.normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);
            }

            logger.info("Sample normalization completed for {} rules.", sample.length);

        } catch (IOException e) {
            logger.error("Failed to load rules from path: {}. Error: {}", config.getRulesPath(), e.getMessage(), e);
        }
    }

    /**
     * Measures the gap between two alternatives based on their score function
     * values and vector differences.
     *
     * @param a The first alternative.
     * @param b The second alternative.
     * @return The computed gap between the two alternatives.
     */
    private double measureGap(IAlternative a, IAlternative b) {
        IAlternative normA = new Alternative(
                normalizer.normalize(a.getVector(), normalizationMethod, false));
        IAlternative normB = new Alternative(
                normalizer.normalize(b.getVector(), normalizationMethod, false));

        double num = abs(scoreFunction.computeScore(normA) - scoreFunction.computeScore(normB));
        double denom = 0d;
        for (int i = 0; i < a.getVector().length; i++) {
            denom += abs(a.getVector()[i] - b.getVector()[i]);
        }
        return num / denom;
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

        // Search for the pair of alternatives with the minimum gap
        double minGap = Double.MAX_VALUE;
        int a1Index = -1;
        int a2Index = -1;

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

                double gap = measureGap(alt1, alt2);
                if (gap < minGap) {
                    minGap = gap;
                    a1Index = iIndex;
                    a2Index = jIndex;
                }
            }
        }

        // If no suitable pair found, select the first two alternatives
        if (a1Index == -1 || a2Index == -1) {
            a1Index = 0;
            a2Index = 1;
        }

        // Prepare the selected alternatives
        IAlternative normA = new Alternative(
                normalizer.normalize(sample[a1Index].getAlternative().getVector(), normalizationMethod, false));
        IAlternative normB = new Alternative(
                normalizer.normalize(sample[a2Index].getAlternative().getVector(), normalizationMethod, false));

        IAlternative[] alternativePair = new IAlternative[] { normA, normB };
        List<DecisionRule> rulePair = new ArrayList<>();
        rulePair.add(sample[a1Index]);
        rulePair.add(sample[a2Index]);

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
}
