package tools.functions.singlevariate;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.functions.multivariate.IMultivariateFunction;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;

public class MultivariateToSinglevariate implements ISinglevariateFunction {
    public @Getter @Setter String Name;

    private @Getter TreeSet<IAlternative[]> history;

    private @Getter HashMap<IAlternative, DecisionRule> seenAlternatives;

    private @Getter LinkedHashMap<IAlternative, DecisionRule> scoreAlternatives;

    private IMultivariateFunction pairwiseUncertainty;

    private @Getter Normalizer normalizer = new Normalizer();

    private @Getter @Setter int maxHistSize = 1000;

    public MultivariateToSinglevariate(String name, IMultivariateFunction pairwiseUncertainty,
            List<DecisionRule> initialRules, int maxHistSize) {
        this.Name = name;
        this.pairwiseUncertainty = pairwiseUncertainty;
        this.maxHistSize = maxHistSize;

        this.history = new TreeSet<>(Comparator.comparingDouble(this::getAlternativeScore).reversed()
                .thenComparingInt(System::identityHashCode));

        this.seenAlternatives = new HashMap<>();
        this.scoreAlternatives = new LinkedHashMap<IAlternative, DecisionRule>(10, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<IAlternative, DecisionRule> eldest) {
                return size() > 10;
            }
        };

        for (DecisionRule rule : initialRules)
            addToHistory(rule.getAlternative(), rule);
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
        scoreAlternatives.put(alternative, rule);

        // Add each new pair of alternatives to the history
        for (IAlternative scoreAlternative : getScoreAlternatives().keySet())
            if (!alternative.equals(scoreAlternative)) {
                getHistory().add(new IAlternative[] { alternative, scoreAlternative });

                if (history.size() > maxHistSize) {
                    history.pollLast();
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

        List<Double> allScores = new ArrayList<>();

        for (IAlternative scoreAlternative : getScoreAlternatives().keySet())
            if (!alternative.equals(scoreAlternative))
                allScores.add(getAlternativeScore(new IAlternative[] { alternative, scoreAlternative }));

        if (allScores.isEmpty()) {
            return 0.0;
        }

        Optional<Double> maxScore = allScores.stream().max(Double::compareTo);
        return maxScore.orElse(-1.0);
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
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
