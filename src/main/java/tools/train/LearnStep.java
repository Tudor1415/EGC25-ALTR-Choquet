package tools.train;

import tools.functions.singlevariate.ISinglevariateFunction;

/**
 * Useful for heuristics to know the current state of the learning algorithm
 */
public interface LearnStep {

    /**
     * Get current score function
     * 
     * @return current score function for this step
     */
    ISinglevariateFunction getCurrentScoreFunction();
}
