package tools.ranking;

import java.util.List;

import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.train.LearnStep;

/**
 * Provide a list of rankings at current learning step
 */
public interface RankingsProvider {

    ISinglevariateFunction getScoreFunction();
    
    List<Ranking<IAlternative>> provideRankings(LearnStep step);
}