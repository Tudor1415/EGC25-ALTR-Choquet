package tools.functions.singlevariate;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.TreeSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.CertaintyFunction;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;
import tools.utils.RuleUtil;

@Getter
@Setter
class AlternativeScore {
    private IAlternative alternative;
    private double score;

    public AlternativeScore(IAlternative alternative, double score) {
        this.alternative = alternative;
        this.score = score;
    }
}

public class MultivariateToSinglevariate implements ISinglevariateFunction {
    public @Getter @Setter String Name;

    private @Getter TreeSet<IAlternative[]> history;

    private @Getter HashMap<IAlternative, DecisionRule> seenAlternatives;

    private @Getter TreeSet<AlternativeScore> scoreAlternatives = new TreeSet<>(
            Comparator.comparingDouble(AlternativeScore::getScore));

    private @Getter @Setter CertaintyFunction pairwiseUncertainty;
    private @Getter ISinglevariateFunction scoreFunction;

    private @Getter Normalizer normalizer = new Normalizer();

    private @Getter @Setter int maxHistSize = 1000;

    public MultivariateToSinglevariate(String name, CertaintyFunction pairwiseUncertainty,
            List<DecisionRule> initialRules, int maxHistSize) {
        this.Name = name;
        this.pairwiseUncertainty = pairwiseUncertainty;
        this.scoreFunction = pairwiseUncertainty.getScoreFunction();

        this.maxHistSize = maxHistSize;

        this.history = new TreeSet<>(Comparator.comparingDouble(this::getAlternativeScore).reversed()
                .thenComparingInt(System::identityHashCode));

        this.seenAlternatives = new HashMap<>();

        for (DecisionRule rule : initialRules)
            addToHistory(rule);
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

    public void addToHistory(DecisionRule rule) {
        DecisionRule copiedRule = RuleUtil.simpleCopy(rule);
        IAlternative alternative = copiedRule.getAlternative();

        // Keep track of the alternatives seen so far
        seenAlternatives.put(alternative, copiedRule);

        double score = getScoreFunction().computeScore(alternative);
        scoreAlternatives.add(
                new AlternativeScore(alternative, score));

        if (scoreAlternatives.size() > 10) {
            scoreAlternatives.pollLast();
        }
        
        // Add each new pair of alternatives to the history
        for (AlternativeScore scoreAlternative : getScoreAlternatives())
            if (!alternative.equals(scoreAlternative.getAlternative())) {
                getHistory().add(new IAlternative[] { alternative, scoreAlternative.getAlternative() });

                if (history.size() > maxHistSize) {
                    history.pollLast();
                }
            }
    }

    @Override
    public double computeScore(DecisionRule rule) {
        addToHistory(rule);
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative) {
        updateNormalization(alternative);

        double score = getScoreFunction().computeScore(alternative);

        AlternativeScore floor = scoreAlternatives.floor(new AlternativeScore(null, score));
        AlternativeScore ceiling = scoreAlternatives.ceiling(new AlternativeScore(null, score));

        AlternativeScore nearest = null;
        if (floor == null) {
            nearest = ceiling;
        } else if (ceiling == null) {
            nearest = floor;
        } else {
            nearest = (score - floor.getScore() <= ceiling.getScore() - score) ? floor : ceiling;
        }

        if (nearest.getAlternative().equals(alternative))
            return 0.0;

        return 1 - pairwiseUncertainty.computeScore(new IAlternative[] { alternative, nearest.getAlternative() });
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        addToHistory(rule);
        return computeScore(alternative);
    }

    private void updateNormalization(IAlternative alternative) {
        getNormalizer().normalize(alternative.getVector(), NormalizationMethod.NO_NORMALIZATION, true);
    }

    public double getAlternativeScore(IAlternative[] alternatives) {
        double[] unNormVector0 = alternatives[0].getVector();
        double[] normVector0 = getNormalizer().normalize(unNormVector0, NormalizationMethod.MIN_MAX_SCALING, false);
        IAlternative normAlternative0 = new Alternative(normVector0);

        double[] unNormVector1 = alternatives[1].getVector();
        double[] normVector1 = getNormalizer().normalize(unNormVector1, NormalizationMethod.MIN_MAX_SCALING, false);
        IAlternative normAlternative1 = new Alternative(normVector1);

        return pairwiseUncertainty.computeScore(new IAlternative[] { normAlternative0, normAlternative1 });
    }
}
