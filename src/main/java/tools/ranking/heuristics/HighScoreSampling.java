package tools.ranking.heuristics;

import java.util.Set;

import experiments.configs.HighScoreSamplingConfig;

import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;
import sampling.SMAS;
import tools.data.Dataset;
import tools.oracles.ArtificialOracle;
import tools.ranking.Ranking;
import tools.ranking.RankingsProvider;
import tools.train.LearnStep;
import tools.utils.RankingUtil;
import tools.rules.DecisionRule;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;

@Getter
@Setter
public class HighScoreSampling implements RankingsProvider {

    // Configurable parameters
    private double noise;
    private int topK;
    private int maximumIterations;
    private NormalizationMethod normalizationMethod;

    // Components
    private ArtificialOracle oracle;
    private SMAS sampler;
    private ISinglevariateFunction scoreFunction;

    // State
    private Set<IAlternative[]> selectedPairs = new HashSet<>();
    private List<Ranking<IAlternative>> rankings = new ArrayList<>();

    public HighScoreSampling(HighScoreSamplingConfig config) {
        this.noise = config.getNoise();
        this.topK = config.getTopK();
        this.maximumIterations = config.getMaximumIterations();
        this.normalizationMethod = config.getNormalizationMethod();
        this.oracle = config.getOracle();

        initializeSampler(config.getDataset(), config.getMeasureNames());
    }

    private void initializeSampler(Dataset dataset, String[] measureNames) {
        ISinglevariateFunction initialFunction = new LinearScoreFunction();
        this.sampler = new SMAS(maximumIterations, dataset, initialFunction, measureNames, topK);
        sampler.setNormalizationTechnique(normalizationMethod);
    }

    @Override
    public List<Ranking<IAlternative>> provideRankings(LearnStep step) {
        // Retrieve the current score function
        scoreFunction = step.getCurrentScoreFunction();

        // Update sampler with the new scoring function
        sampler.setScoringFunction(scoreFunction);

        // Sample new alternatives
        List<DecisionRule> sample = sampler.sample();
        Normalizer normalizer = sampler.getNormalizer();

        // Normalize sampled alternatives
        Alternative normalized0 = new Alternative(
                normalizer.normalize(sample.get(0).getAlternative().getVector(), normalizationMethod, false));
        Alternative normalized1 = new Alternative(
                normalizer.normalize(sample.get(1).getAlternative().getVector(), normalizationMethod, false));

        // Add selected pair
        IAlternative[] alternativePair = new IAlternative[]{normalized0, normalized1};
        selectedPairs.add(alternativePair);

        // Compute rankings using the oracle
        if (oracle instanceof ArtificialOracle) {
            if (noise == 0) {
                rankings.add(RankingUtil.computeRankingWithOracle(oracle, sample, alternativePair));
            } else {
                rankings.add(RankingUtil.computeNoisyRankingWithOracle((ArtificialOracle) oracle, sample, noise));
            }
        } else {
            throw new IllegalArgumentException("Unsupported Oracle type: " + oracle.getClass().getName());
        }

        return rankings;
    }
}
