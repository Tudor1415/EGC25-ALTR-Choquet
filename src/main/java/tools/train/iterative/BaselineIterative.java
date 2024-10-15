package tools.train.iterative;

import java.util.List;

import tools.alternatives.IAlternative;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.ranking.Ranking;
import tools.ranking.RankingsProvider;
import tools.train.IterativeRankingLearn;

/**
 * The baseline model assigns equal weights to all measures, and the learning
 * process involves setting up the function parameters accordingly. In other
 * words, the weights remain constant during the learning process.
 *
 * @see IterativeRankingLearn
 * @param nbIterations     The number of iterations for the learning process.
 * @param rankingsProvider The provider of rankings used for learning.
 * @param func             The scoring function used in the learning process.
 * @param nbMeasures       The number of measures or criteria.
 */
public class BaselineIterative extends IterativeRankingLearn {

    /**
     * Array representing the weights assigned to each measure in the baseline
     * model.
     */
    private double[] weights;

    public BaselineIterative(int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func,
            int nbMeasures) {
        super(nbIterations, rankingsProvider, func, nbMeasures);
        weights = new double[nbMeasures];
        for (int i = 0; i < nbMeasures; i++) {
            weights[i] = 1d; // Assign equal weights to all measures initially
        }
    }

    /**
     * Learns the function parameters for the baseline model based on the provided
     * rankings.
     * In the baseline model, all measures have equal weights, and the function type
     * is linear.
     *
     * @param rankings The list of rankings used for learning.
     * @return FunctionParameters representing the learned function parameters for
     *         the baseline model.
     * @throws Exception If an error occurs during the learning process.
     */
    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        FunctionParameters params = new FunctionParameters();
        params.setFunctionType(LinearScoreFunction.TYPE);
        params.setWeights(weights);
        params.setNbCriteria(nbMeasures);
        return params;
    }
}
