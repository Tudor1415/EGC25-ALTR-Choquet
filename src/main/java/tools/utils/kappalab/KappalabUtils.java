package tools.utils.kappalab;

import tools.alternatives.IAlternative;
import tools.ranking.Ranking;

/**
 * Utility class for working with Kappalab.
 */
public class KappalabUtils {
    /**
     * Adds preferences from a ranking to the Kappalab input.
     *
     * @param ranking The ranking of alternatives.
     * @param input   The Kappalab input to which preferences will be added.
     * @param delta   The delta value indicating the preference strength, where a
     *                positive value means a preference for 'a' over 'b',
     *                and a negative value means a preference for 'b' over 'a'.
     */
    public static void addRankingToKappalabInput(Ranking<IAlternative> ranking, KappalabInput input, double delta) {
        IAlternative[] rankAlternatives = ranking.getObjects();
        for (int i = 0; i < rankAlternatives.length - 1; i++) {
            input.addPreference(rankAlternatives[i], rankAlternatives[i+1], delta, ranking.getScores());
        }
    }

}
