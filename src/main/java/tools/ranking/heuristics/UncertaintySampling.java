package tools.ranking.heuristics;

import lombok.Getter;
import lombok.Setter;
import sampling.MMAS;
import tools.data.Dataset;
import tools.ranking.Ranking;
import tools.train.LearnStep;
import tools.utils.RankingUtil;
import tools.rules.DecisionRule;
import tools.alternatives.Alternative;
import tools.normalization.Normalizer;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.alternatives.IAlternative;
import experiments.configs.UncertaintySamplingConfig;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.multivariate.outRankingCertainties.Thurstone;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;

import java.util.Set;
import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;

@Getter
@Setter
public class UncertaintySampling implements RankingsProvider {

    // Configurable parameters
    private double noise;
    private int maximumIterations;
    private String certaintyType;
    private NormalizationMethod normalizationMethod;
    private int nbLearningIterations;
    private int changePeriod;

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;

    // Internal components
    private CertaintyFunction pairwiseCertaintyFunction;
    private ISinglevariateFunction scoreFunction;
    private MMAS sampler;

    // State
    private Set<IAlternative[]> selectedPairs = new HashSet<>();
    private List<Ranking<IAlternative>> rankings = new ArrayList<>();
    private int learningIteration = 0;

    public UncertaintySampling(UncertaintySamplingConfig config) {
        // Set configurable parameters
        this.noise = config.getNoise();
        this.maximumIterations = config.getMaximumIterations();
        this.certaintyType = config.getCertaintyType();
        this.normalizationMethod = config.getNormalizationMethod();
        this.nbLearningIterations = config.getNbLearningIterations();

        // Set required components
        this.oracle = config.getOracle();
        this.dataset = config.getDataset();
        this.measureNames = config.getMeasureNames();

        // Initialize components
        this.changePeriod = config.getChangePeriod();
        this.scoreFunction = new LinearScoreFunction();
        initializeCertaintyFunction();
        initializeSampler(config.isStartUncertainty());
    }

    private void initializeCertaintyFunction() {
        switch (certaintyType) {
            case "ScoreDifference":
                this.pairwiseCertaintyFunction = new PairwiseUncertainty("ScoreDifferencePairUncertainty",
                        new ScoreDifference(scoreFunction));
                break;
            case "BradleyTerry":
                this.pairwiseCertaintyFunction = new PairwiseUncertainty("BradleyTerryPairUncertainty",
                        new BradleyTerry(scoreFunction));
                break;
            case "Thurstone":
                this.pairwiseCertaintyFunction = new PairwiseUncertainty("ThurstonePairUncertainty",
                        new Thurstone(scoreFunction));
                break;
            default:
                // Default to ScoreDifference if invalid type is provided
                this.pairwiseCertaintyFunction = new PairwiseUncertainty("ScoreDifferencePairUncertainty",
                        new ScoreDifference(scoreFunction));
                break;
        }
    }

    private void initializeSampler(boolean isStartUncertainty) {
        this.sampler = new MMAS(maximumIterations, 1, dataset, pairwiseCertaintyFunction, measureNames);
        this.sampler.setNormalizationTechnique(normalizationMethod);
        this.sampler.setUncertainty(isStartUncertainty);
    }

    @Override
    public List<Ranking<IAlternative>> provideRankings(LearnStep step) {
        // Increment the learning iteration counter
        learningIteration++;

        // Update score function
        this.scoreFunction = step.getCurrentScoreFunction();

        // Update certainty function with the new score function
        initializeCertaintyFunction();

        // Update sampler with the new scoring function
        this.sampler.setScoringFunction(scoreFunction);

        // Change sampler behavior after CHANGE_PERIOD iterations
        if (learningIteration > changePeriod && learningIteration % changePeriod == 0)
            sampler.setUncertainty(!sampler.isUncertainty());

        // Sample new alternatives from the dataset
        List<DecisionRule[]> samplePairs = sampler.sample();

        // Assuming that sampler.sample() returns a list with pairs of DecisionRule[]
        if (samplePairs.isEmpty()) {
            throw new IllegalStateException("Sampler returned no samples");
        }

        // Process each sampled pair (here, we assume only one pair is sampled per iteration)
        DecisionRule[] selectedPairRules = samplePairs.get(0);

        List<DecisionRule> listSample = new ArrayList<>();
        listSample.add(selectedPairRules[0]);
        listSample.add(selectedPairRules[1]);

        Normalizer normalizer = sampler.getNormalizer();

        // Normalize sampled alternatives
        IAlternative normalized0 = new Alternative(
                normalizer.normalize(selectedPairRules[0].getAlternative().getVector(), normalizationMethod, false));
        IAlternative normalized1 = new Alternative(
                normalizer.normalize(selectedPairRules[1].getAlternative().getVector(), normalizationMethod, false));

        // Add the selected pair to the set of selected pairs
        IAlternative[] alternativePair = new IAlternative[]{normalized0, normalized1};
        selectedPairs.add(alternativePair);

        // Compute the ranking for the selected pair using the oracle
        if (noise == 0) {
            rankings.add(RankingUtil.computeRankingWithOracle(oracle, listSample, alternativePair));
        } else {
            rankings.add(RankingUtil.computeNoisyRankingWithOracle(oracle, listSample, noise));
        }

        // Return all the computed rankings (from all prior iterations including this one)
        return rankings;
    }
}
