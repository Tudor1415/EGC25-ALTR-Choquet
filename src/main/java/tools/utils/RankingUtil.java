package tools.utils;

import java.util.List;
import java.util.stream.IntStream;

import tools.alternatives.IAlternative;
import tools.oracles.ArtificialOracle;
import tools.ranking.Ranking;
import tools.rules.DecisionRule;

/**
 * Utility class for working with rankings.
 */
public class RankingUtil {
        /**
         * Computes a ranking of alternatives based on the provided oracle comparator.
         * The ranking is determined by sorting the indices of alternatives according to
         * the oracle's comparison results.
         *
         * @param oracle       The comparator used for determining the ranking order.
         * @param alternatives The array of alternatives to be ranked.
         * @return A Ranking object representing the computed ranking.
         */
        public static Ranking<IAlternative> computeRankingWithOracle(ArtificialOracle oracle, List<DecisionRule> rules,
                        IAlternative[] normalizedAlternatives) {

                int[] ranking = IntStream.range(0, rules.size())
                                .boxed()
                                // Sort the indices based on the scores computed by the oracle
                                .sorted((i, j) -> oracle.compare(rules.get(i), rules.get(j)))
                                .mapToInt(i -> i)
                                .toArray();

                // Initialize the alternativesArray based on the sorted rules
                IAlternative[] alternativesArray = new IAlternative[] { normalizedAlternatives[ranking[0]],
                                normalizedAlternatives[ranking[1]] };

                // Compute the oracle scores
                Double[] scores = new Double[] { oracle.computeScore(rules.get(ranking[0])),
                                oracle.computeScore(rules.get(ranking[1])) };

                // Return the final Ranking object
                return new Ranking<>(alternativesArray, scores);
        }

        public static Ranking<IAlternative> computeNoisyRankingWithOracle(ArtificialOracle oracle,
                        List<DecisionRule> rules, double noise) {

                // Create an array to hold the ranking indices
                int[] ranking = IntStream.range(0, rules.size())
                                .boxed()
                                // Sort the indices based on the scores computed by the oracle's scoring
                                // function.
                                .sorted((i, j) -> oracle.compareNoisy(rules.get(i), rules.get(j), noise))
                                .mapToInt(i -> i)
                                .toArray();

                IAlternative[] alternativesArray = rules.stream()
                                .sorted((rule1, rule2) -> oracle.compare(rule1, rule2))
                                .map(DecisionRule::getAlternative)
                                .toArray(IAlternative[]::new);

                // Compute the oracle scores
                Double[] scores = new Double[] { oracle.computeScore(rules.get(ranking[0])),
                                oracle.computeScore(rules.get(ranking[1])) };

                // TODO: give the normalized alternatives
                // Create and return a new Ranking object with the computed ranking
                return new Ranking<>(alternativesArray, scores);
        }
}
