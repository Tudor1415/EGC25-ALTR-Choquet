package tools.train;

import java.util.List;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.ranking.Ranking;
import tools.ranking.RankingsProvider;
import tools.utils.FunctionUtil;

/**
 * Abstract class representing an iterative ranking learning strategy.
 * This class provides a framework for implementing iterative ranking learning
 * algorithms.
 * It extends the {@link AbstractRankingLearning} class and implements the
 * {@link LearnStep} interface.
 * Concrete subclasses must implement the {@link #learn(List)} method to define
 * the learning process.
 * The class maintains information about the number of iterations, rankings
 * provider, and the current scoring function.
 *
 * @see AbstractRankingLearning
 * @see LearnStep
 */
public abstract class IterativeRankingLearn extends AbstractRankingLearning implements LearnStep {

    @Getter
    @Setter
    private String name;

    /**
     * The number of iterations for the learning process.
     */
    private int nbIterations;

    /**
     * The provider of rankings used for learning.
     */
    private RankingsProvider rankingsProvider;

    /**
     * The scoring function used in the learning process.
     */
    private ISinglevariateFunction func;

    /**
     * The time remaining in milliseconds for the learning process.
     */
    protected long timeRemaining;

    /**
     * Constructs an IterativeRankingLearn instance with the specified parameters.
     *
     * @param nbIterations     The number of iterations for the learning process.
     * @param rankingsProvider The provider of rankings used for learning, usually a
     *                         heuristic like Min Gaps.
     * @param func             The scoring function used in the learning process
     *                         (eg. Choquet integral).
     * @param nbMeasures       The number of measures or criteria.
     */
    public IterativeRankingLearn(int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func,
            int nbMeasures) {
        this.nbIterations = nbIterations;
        this.rankingsProvider = rankingsProvider;
        this.func = func;
        this.nbMeasures = nbMeasures;
    }

    /**
     * Learns the function parameters iteratively based on the provided rankings.
     * This method implements the iterative learning process, updating function
     * parameters for each iteration.
     * 
     * @return FunctionParameters representing the learned function parameters.
     * @throws Exception If an error occurs during the learning process.
     */
    public FunctionParameters learn() throws Exception {
        // Initialize the function parameters
        FunctionParameters params = new FunctionParameters();
        params.setNbIterations(nbIterations);

        // During the last iteration we skip the training and only do the testing
        for (int i = 0; i < nbIterations; i++) {
            // Accumulate the time spent on learning

            double timeToLearn = params.getTimeToLearn();

            // Learn based on provided rankings (implemented by the subclass)
            params = learnFromRankings(rankingsProvider.provideRankings(this));

            // Update the total learning time
            params.setTimeToLearn(timeToLearn + params.getTimeToLearn());

            // Check for errors during the learning process
            if (params.getErrorMessages() != null) {
                params.setNbIterations(i + 1);
                return params;
            }

            // Update the scoring function based on the learned parameters
            func = FunctionUtil.getScoreFunction(params);

            support.firePropertyChange("func", null, func);
        }

        return params;
    }

    /**
     * Learns the function parameters based on the provided rankings.
     * Concrete subclasses must implement this method to define the learning
     * process.
     *
     * @param rankings The list of rankings used for learning.
     * @return FunctionParameters representing the learned function parameters.
     * @throws Exception If an error occurs during the learning process.
     */
    public abstract FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception;

    /**
     * Retrieves the current scoring function used in the learning process
     * (e.g. the state of the Choquet integral at iteration k).
     * 
     * @return The current scoring function.
     */
    public ISinglevariateFunction getCurrentScoreFunction() {
        return func;
    }

    /**
     * Sets the time limit for the iterative learning process. Overrides the parent
     * method
     * to update the remaining time in milliseconds based on the specified limit.
     *
     * @param timeLimit The time limit in seconds for the iterative learning
     *                  process.
     */
    @Override
    public void setTimeLimit(int timeLimit) {
        // Set the time limit using the parent method
        super.setTimeLimit(timeLimit);

        // Update the remaining time in milliseconds
        timeRemaining = timeLimit * 1000L;
    }
}
