package tools.ranking.heuristics;

import java.util.Set;
import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;

import lombok.Getter;
import lombok.Setter;
import sampling.MMAS;
import tools.data.Dataset;
import tools.oracles.Oracle;
import tools.ranking.Ranking;
import tools.train.LearnStep;
import tools.utils.RankingUtil;
import tools.rules.DecisionRule;
import tools.alternatives.Alternative;
import tools.normalization.Normalizer;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.multivariate.outRankingCertainties.Thurstone;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;

public class UncertaintySampling implements RankingsProvider {

    private @Getter @Setter double noise;

    // The oracle used to rank the selected pair of rules
    private ArtificialOracle oracle;

    // The uncertainty used for sampling
    private @Getter CertaintyFunction pairwiseCertaintyFunction;

    // The state of the approximation function at the current iteration
    private @Getter ISinglevariateFunction scoreFunction;

    // The sampling instance used to sample the rules
    private MMAS sampler;

    // The maximum iterations used for sampling
    private @Setter @Getter int maximum_iterations = MAXIMUM_ITERATIONS;

    // Number of learning iterations
    private @Getter @Setter int nbLearningIteration;
    private int learningIteration;

    // The list of all selected pairs of alternatives from all the iterations
    // and their respective ranking given by the oracle.
    private Set<IAlternative[]> selectedPairs = new HashSet<>();
    private List<Ranking<IAlternative>> rankings = new ArrayList<>();

    private static final double DEFAULT_NOISE = 0d;
    private static final int MAXIMUM_ITERATIONS = 100;
    private static final int CHANGE_PERIOD = 100;

    public UncertaintySampling(ArtificialOracle oracle, Dataset dataset, String[] measureNames, double noise,
            int nbLearningIteration) {
        this(oracle, dataset, measureNames, noise, MAXIMUM_ITERATIONS, "ScoreDifference",
                NormalizationMethod.MIN_MAX_SCALING, nbLearningIteration);
    }

    public UncertaintySampling(ArtificialOracle oracle, Dataset dataset, String[] measureNames,
            int nbLearningIteration) {
        this(oracle, dataset, measureNames, DEFAULT_NOISE, MAXIMUM_ITERATIONS, "ScoreDifference",
                NormalizationMethod.MIN_MAX_SCALING, nbLearningIteration);
    }

    public UncertaintySampling(ArtificialOracle oracle, Dataset dataset, String[] measureNames, String certaintyType,
            int nbLearningIteration) {
        this(oracle, dataset, measureNames, DEFAULT_NOISE, MAXIMUM_ITERATIONS, certaintyType,
                NormalizationMethod.MIN_MAX_SCALING, nbLearningIteration);
    }

    public UncertaintySampling(ArtificialOracle oracle, Dataset dataset, String[] measureNames, double noise,
            int maximumIterations, String certaintyType, NormalizationMethod normalizationMethod,
            int nbLearningIteration) {
        this.oracle = oracle;
        this.noise = noise;
        this.nbLearningIteration = nbLearningIteration;

        updateCertaintyFunction(certaintyType, new LinearScoreFunction());
        initializeSampler(dataset, measureNames, maximumIterations);
        setSamplerNormalizationMethod(normalizationMethod);
    }

    public void updateCertaintyFunction(String certaintyType, ISinglevariateFunction scoreFunction) {
        this.scoreFunction = scoreFunction;

        switch (certaintyType) {
            case "ScoreDifferece":
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
                this.pairwiseCertaintyFunction = new PairwiseUncertainty("ScoreDifferencePairUncertainty",
                        new ScoreDifference(scoreFunction));
        }
    }

    private void initializeSampler(Dataset dataset, String[] measureNames, int maximumIterations) {
        this.sampler = new MMAS(MAXIMUM_ITERATIONS, 1, dataset, pairwiseCertaintyFunction, measureNames);
    }

    /**
     * The algorithm selects pairs of alternatives with minimal gaps in their score
     * and generates rankings for each pair.
     * The process is repeated until the desired number of rankings is obtained.
     *
     * @param step The current iteration step containing the score function and
     *             other information.
     * @return A list of rankings generated by the MinGapsRankingsProvider
     *         algorithm.
     */
    @Override
    public List<Ranking<IAlternative>> provideRankings(LearnStep step) {
        // Increment the learning iteration counter
        learningIteration++;

        // Retrieving the state of the approximation function at the current iteration
        scoreFunction = step.getCurrentScoreFunction();

        // Sampling new rules using the sampler with the updated approximation function
        sampler.setScoringFunction(scoreFunction);

        // if (learningIteration > CHANGE_PERIOD && learningIteration % CHANGE_PERIOD == 0)
        if (learningIteration > CHANGE_PERIOD)
            sampler.setUncertainty(true);

        // Sample new alternatives from the test dataset
        List<DecisionRule[]> sample = sampler.sample();

        List<DecisionRule> listSample = new ArrayList<>();
        listSample.add(sample.get(0)[0]);
        listSample.add(sample.get(0)[1]);

        Normalizer normalizer = sampler.getNormalizer();

        Alternative normalized0 = new Alternative(normalizer.normalize(listSample.get(0).getAlternative().getVector(),
                NormalizationMethod.MIN_MAX_SCALING, false));
        Alternative normalized1 = new Alternative(normalizer.normalize(listSample.get(1).getAlternative().getVector(),
                NormalizationMethod.MIN_MAX_SCALING, false));

        // Add the selected pair to the set of selected pairs
        IAlternative[] alternativePair = new IAlternative[] { normalized0, normalized1 };
        selectedPairs.add(alternativePair);

        // Compute the ranking for the selected pair using the oracle
        if (oracle instanceof ArtificialOracle) {
            if (getNoise() == 0) {
                rankings.add(RankingUtil.computeRankingWithOracle(oracle, listSample, alternativePair));
            } else {
                rankings.add(RankingUtil.computeNoisyRankingWithOracle(oracle, listSample, getNoise()));
            }
        } else if (oracle instanceof Oracle) {
            rankings.add(RankingUtil.computeRankingWithOracle(oracle, listSample, alternativePair));
        } else {
            throw new IllegalArgumentException("Unsupported Oracle type: " + oracle.getClass().getName());
        }

        // Return all the computed rankings (from all the prior iterations including
        // this one)
        return rankings;
    }

    public void setSamplerNormalizationMethod(NormalizationMethod normalizationMethod) {
        sampler.setNormalizationTechnique(normalizationMethod);
    }
}