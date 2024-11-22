package tools.utils;

import java.util.List;
import java.util.stream.IntStream;

import tools.ranking.Ranking;
import tools.rules.DecisionRule;
import tools.oracles.ArtificialOracle;
import tools.alternatives.IAlternative;

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
        public static Ranking<IAlternative> computeRankingWithOracle(
                        ArtificialOracle oracle,
                        List<DecisionRule> rules,
                        IAlternative[] normalizedAlternatives) {

                // Step 1: Compute the ranking of rules based on oracle's comparison
                int[] ranking = IntStream.range(0, rules.size())
                                .boxed()
                                .sorted((i, j) -> oracle.compare(rules.get(i), rules.get(j)))
                                .mapToInt(i -> i)
                                .toArray();

                // Step 2: Extract the top-ranked alternatives based on the computed ranking
                int topRankIndex = ranking[0];
                int secondRankIndex = ranking[1];

                IAlternative topAlternative = normalizedAlternatives[topRankIndex];
                IAlternative secondAlternative = normalizedAlternatives[secondRankIndex];
                IAlternative[] alternativesArray = new IAlternative[] { topAlternative, secondAlternative };

                // Step 3: Compute the oracle scores for the top-ranked rules
                double topScore = oracle.computeScore(rules.get(topRankIndex));
                double secondScore = oracle.computeScore(rules.get(secondRankIndex));
                Double[] scores = new Double[] { topScore, secondScore };

                // Step 4: Return the final Ranking object
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
