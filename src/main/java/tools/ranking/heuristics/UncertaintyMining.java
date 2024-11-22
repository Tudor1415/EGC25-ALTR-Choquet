package tools.ranking.heuristics;

import experiments.configs.MiningConfig;
import experiments.configs.MiningConfig;
import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.oracles.Oracle;
import tools.ranking.Ranking;
import tools.ranking.RankingsProvider;
import tools.rules.DecisionRule;
import tools.train.LearnStep;
import tools.utils.RankingUtil;
import tools.utils.RandomUtil;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static java.lang.Math.abs;

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
        if (config.getSample() != null) {
            this.sample = config.getSample();
        } else if (config.getRulesPath() != null) {
            // Load sample from rules file
            this.sample = loadRulesFromFile(config.getRulesPath());
        } else {
            // Generate sample from dataset
            int sampleSize = config.getSampleSize();
            if (sampleSize <= 0) {
                throw new IllegalArgumentException("Sample size must be greater than 0");
            }
            this.sample = sampleRulesFromDataset(sampleSize);
        }

        // Normalize the sample
        for (DecisionRule rule : sample) {
            normalizer.normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);
        }
    }

    private DecisionRule[] loadRulesFromFile(String rulesPath) {
        // Implement loading rules from a file
        // Placeholder implementation:
        // DecisionRule[] rules = ...;
        // return rules;
        throw new UnsupportedOperationException("Loading rules from file not implemented yet");
    }

    private DecisionRule[] sampleRulesFromDataset(int sampleSize) {
        // Implement sampling rules from the dataset
        // Placeholder implementation:
        // You might use a sampling method to generate DecisionRule[] from the dataset
        // For now, we'll throw an exception
        throw new UnsupportedOperationException("Sampling rules from dataset not implemented yet");
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
                if (selectedPairs.contains(new IAlternative[]{alt1, alt2})) {
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

        IAlternative[] alternativePair = new IAlternative[]{normA, normB};
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
