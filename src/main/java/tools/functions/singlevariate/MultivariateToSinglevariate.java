package tools.functions.singlevariate;

import java.util.List;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Comparator;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.utils.AlternativeScore;
import tools.alternatives.Alternative;
import tools.normalization.Normalizer;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.normalization.Normalizer.NormalizationMethod;

public class MultivariateToSinglevariate implements ISinglevariateFunction {
    public @Getter @Setter String Name;

    private @Getter TreeSet<IAlternative[]> history;

    private @Getter HashMap<IAlternative, DecisionRule> seenAlternatives;

    private @Getter TreeSet<AlternativeScore> scoreAlternatives;

    private LinkedList<AlternativeScore> insertionOrder = new LinkedList<>();

    private CertaintyFunction pairwiseUncertainty;

    private @Getter Normalizer normalizer = new Normalizer();

    private @Getter @Setter int maxHistSize = 100;

    private @Getter @Setter int nbOfScoringAlternatives = 10;

    public MultivariateToSinglevariate(String name, CertaintyFunction pairwiseUncertainty,
            List<DecisionRule> initialRules, int maxHistSize) {
        this.Name = name;
        this.pairwiseUncertainty = pairwiseUncertainty;
        this.maxHistSize = maxHistSize;

        this.history = new TreeSet<IAlternative[]>(Comparator
                .<IAlternative[]>comparingDouble(pair -> getAlternativePairScore(pair))
                .reversed()
                .thenComparingInt(System::identityHashCode));

        this.seenAlternatives = new HashMap<>();
        this.scoreAlternatives = new TreeSet<>((as1, as2) -> {
            // Compare alternatives first for uniqueness
            if (as1.getAlternative().equals(as2.getAlternative())) {
                return 0; // Same alternative, treat as duplicate
            }

            // If alternatives are different, impose a consistent order
            // Use the score as a secondary ordering criterion
            int scoreCompare = Double.compare(as1.getScore(), as2.getScore());
            if (scoreCompare != 0) {
                return scoreCompare;
            }

            // If both score and alternative comparison fail, impose arbitrary order
            return Integer.compare(System.identityHashCode(as1.getAlternative()),
                    System.identityHashCode(as2.getAlternative()));
        });

        for (DecisionRule rule : initialRules) {
            IAlternative alternative = rule.getAlternative();
            seenAlternatives.put(alternative, rule);
    
            updateNormalization(alternative);
    
            double score = getAlternativeScore(alternative);
            AlternativeScore altScore = new AlternativeScore(alternative, score, rule);
    
            if (scoreAlternatives.size() < nbOfScoringAlternatives) {
                scoreAlternatives.add(altScore);
                insertionOrder.addLast(altScore);
            }
        }
    }

    public List<DecisionRule[]> getTopK(int k) {
        List<DecisionRule[]> topKRules = new ArrayList<>();
        int count = 0;

        for (IAlternative[] alternativePair : history) {
            if (count >= k) {
                break;
            }

            DecisionRule rule1 = seenAlternatives.get(alternativePair[0]);
            DecisionRule rule2 = seenAlternatives.get(alternativePair[1]);

            if (rule1 != null && rule2 != null) {
                topKRules.add(new DecisionRule[] { rule1, rule2 });
            }

            count++;
        }

        return topKRules;
    }

    public void addToHistory(IAlternative alternative, DecisionRule rule) {
        // Keep track of the alternatives seen so far
        seenAlternatives.put(alternative, rule);

        // Compute the score and check for NaN
        double score = getAlternativeScore(alternative);
        if (Double.isNaN(score)) {
            return;
        }

        // Create an AlternativeScore
        AlternativeScore altScore = new AlternativeScore(alternative, score, rule);

        // Add to scoreAlternatives (TreeSet for score sorting)
        scoreAlternatives.add(altScore);

        // Add to insertionOrder (queue for oldest tracking)
        insertionOrder.addLast(altScore);

        // Enforce size limit by removing the oldest alternative
        if (insertionOrder.size() > nbOfScoringAlternatives) {
            AlternativeScore oldest = insertionOrder.removeFirst(); // Remove the oldest alternative
            scoreAlternatives.remove(oldest); // Also remove it from the TreeSet
        }

        // Add each new pair of alternatives to the history
        for (AlternativeScore scoreAlt : scoreAlternatives) {
            IAlternative scoreAlternative = scoreAlt.getAlternative();
            if (!alternative.equals(scoreAlternative)) {
                getHistory().add(new IAlternative[] { alternative, scoreAlternative });

                if (history.size() > maxHistSize) {
                    history.pollLast();
                }
            }
        }
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative) {
        updateNormalization(alternative);

        // Compute the score for the alternative directly
        double score = getAlternativeScore(alternative);

        AlternativeScore altScore = new AlternativeScore(alternative, score, null);

        // Find floor and ceiling in the scoreAlternatives TreeSet
        AlternativeScore floor = scoreAlternatives.floor(altScore);
        AlternativeScore ceiling = scoreAlternatives.ceiling(altScore);

        AlternativeScore nearest = null;

        if (floor == null && ceiling == null) {
            // No other samples exist; set default gapScore
            nearest = altScore;
        } else if (floor == null) {
            nearest = ceiling;
        } else if (ceiling == null) {
            nearest = floor;
        } else {
            nearest = (score - floor.getScore() <= ceiling.getScore() - score) ? floor : ceiling;
        }

        double gapScore = 0.0;
        if (nearest != null && nearest != altScore) {
            double gap = Math.abs(score - nearest.getScore());
            gapScore = 1.0 - gap;
        }

        return gapScore;
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(alternative);
    }

    private void updateNormalization(IAlternative alternative) {
        getNormalizer().normalize(alternative.getVector(), NormalizationMethod.NO_NORMALIZATION, true);
    }

    public double getAlternativePairScore(IAlternative[] alternatives) {
        double[] unNormVector0 = alternatives[0].getVector();
        double[] normVector0 = getNormalizer().normalize(unNormVector0, NormalizationMethod.MIN_MAX_SCALING, false);
        IAlternative normAlternative0 = new Alternative(normVector0);

        double[] unNormVector1 = alternatives[1].getVector();
        double[] normVector1 = getNormalizer().normalize(unNormVector1, NormalizationMethod.MIN_MAX_SCALING, false);
        IAlternative normAlternative1 = new Alternative(normVector1);

        return pairwiseUncertainty.computeScore(new IAlternative[] { normAlternative0, normAlternative1 });
    }

    public double getAlternativeScore(IAlternative alternative) {
        double[] unNormVector = alternative.getVector();
        double[] normVector = getNormalizer().normalize(unNormVector, NormalizationMethod.MIN_MAX_SCALING, false);
        IAlternative normAlternative = new Alternative(normVector);

        return pairwiseUncertainty.getScoreFunction().computeScore(normAlternative);
    }
}
