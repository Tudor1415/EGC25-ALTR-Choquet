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

    private boolean isUncertainty = true;

    public MultivariateToSinglevariate(String name, CertaintyFunction pairwiseUncertainty,
            List<DecisionRule> initialRules, int maxHistSize, boolean isUncertainty) {
        this.Name = name;
        this.pairwiseUncertainty = pairwiseUncertainty;
        this.maxHistSize = maxHistSize;
        setIsUncertainty(isUncertainty);

        this.seenAlternatives = new HashMap<>();
        this.scoreAlternatives = new TreeSet<>((as1, as2) -> {
            if (as1.getAlternative().equals(as2.getAlternative())) {
                return 0;
            }

            int scoreCompare = Double.compare(as1.getScore(), as2.getScore());
            if (scoreCompare != 0) {
                return scoreCompare;
            }

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

    public boolean getIsUncertainty() {
        return isUncertainty;
    }

    public void setIsUncertainty(boolean isUncertainty) {
        this.isUncertainty = isUncertainty;
        TreeSet<IAlternative[]> newHistory = new TreeSet<>(getComparatorForHistory());
        if (this.history != null) {
            newHistory.addAll(this.history);
        }
        this.history = newHistory;
    }

    private Comparator<IAlternative[]> getComparatorForHistory() {
        Comparator<IAlternative[]> baseComparator = Comparator
                .<IAlternative[]>comparingDouble(this::getAlternativePairScore)
                .thenComparingInt(System::identityHashCode);
        return isUncertainty ? baseComparator.reversed() : baseComparator;
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
        seenAlternatives.put(alternative, rule);

        double score = getAlternativeScore(alternative);
        if (Double.isNaN(score)) {
            return;
        }

        AlternativeScore altScore = new AlternativeScore(alternative, score, rule);

        scoreAlternatives.add(altScore);
        insertionOrder.addLast(altScore);

        if (insertionOrder.size() > nbOfScoringAlternatives) {
            AlternativeScore oldest = insertionOrder.removeFirst();
            scoreAlternatives.remove(oldest);
        }

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

        double score = getAlternativeScore(alternative);

        AlternativeScore altScore = new AlternativeScore(alternative, score, null);

        AlternativeScore floor = scoreAlternatives.floor(altScore);
        AlternativeScore ceiling = scoreAlternatives.ceiling(altScore);

        AlternativeScore nearest = null;

        if (floor == null && ceiling == null) {
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
            gapScore = isUncertainty ? (1.0 - gap) : gap;
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
