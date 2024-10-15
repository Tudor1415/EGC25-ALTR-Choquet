package sampling;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;
import tools.functions.multivariate.outRankingCertainties.Thurstone;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;
import tools.utils.RuleUtil;

public class SMAS implements ISampler {
    private static final double DEFAULT_SMOOTH_COUNTS = 1e-6d;

    private @Getter @Setter int maximumIterations;
    private @Getter @Setter int topK = 1;
    private @Getter @Setter DecisionRule rule;
    private @Getter @Setter Dataset dataset;
    private @Getter @Setter TreeSet<DecisionRule> topRules;
    private @Getter @Setter CertaintyFunction outRankingCertainty;
    private @Getter ISinglevariateFunction scoringFunction;
    private @Getter @Setter String[] measureNames;
    private @Getter @Setter double smoothCounts = 1e-6d;
    private @Getter RandomUtil random = new RandomUtil();
    private @Getter List<Double> scoreHistory = new ArrayList<>();
    private @Getter @Setter Normalizer.NormalizationMethod normalizationTechnique = NormalizationMethod.MIN_MAX_SCALING;
    private @Getter Normalizer normalizer = new Normalizer();

    public SMAS(int maximumIterations, Dataset dataset, CertaintyFunction outRankingCertainty,
            ISinglevariateFunction scoringFunction, String[] measureNames, double smoothCounts, int topK) {
        this.maximumIterations = maximumIterations;
        this.dataset = dataset;
        this.outRankingCertainty = outRankingCertainty;
        this.scoringFunction = scoringFunction;
        this.measureNames = measureNames;
        this.smoothCounts = smoothCounts;
        this.topK = topK;
        this.topRules = new TreeSet<>(Comparator.comparingDouble(this::getValidRuleScore).reversed()
                .thenComparingInt(System::identityHashCode));
    }

    public SMAS(int maximumIterations, Dataset dataset, ISinglevariateFunction scoringFunction, String[] measureNames,
            int topK) {
        this(maximumIterations, dataset, new BradleyTerry(scoringFunction), scoringFunction, measureNames,
                DEFAULT_SMOOTH_COUNTS, topK);
    }

    @Override
    public List<DecisionRule> sample() {
        initNormalization();
        DecisionRule initialRule = getDataset().getRandomValidRules(1, smoothCounts, measureNames).get(0);
        setRule(initialRule);

        // Erase the memory before each run
        topRules = new TreeSet<>(Comparator.comparingDouble(this::getValidRuleScore).reversed());
        
        topRules.add(RuleUtil.simpleCopy(getRule()));

        for (int i = 0; i < getMaximumIterations(); i++) {
            setRule(updateRule(getRule()));
            double currentScore = getValidRuleScore(getRule());
            scoreHistory.add(currentScore);

            if (!topRules.contains(getRule())) {
                topRules.add(RuleUtil.simpleCopy(getRule()));

                if (topRules.size() > topK) {
                    topRules.pollLast();
                }
            }
        }

        // Expand the rules so they contain all the required data
        List<DecisionRule> finalRules = new ArrayList<>();
        for (DecisionRule rule : topRules) {
            rule.expandSimpleCopy(initialRule);
            finalRules.add(rule);
        }

        return finalRules;
    }

    @Override
    public double getValidRuleScore(DecisionRule rule) {
        if (RuleUtil.isValid(rule)) {
            double[] unNormVector = rule.getAlternative().getVector();
            double[] normVector = getNormalizer().normalize(unNormVector, getNormalizationTechnique(), false);
            IAlternative normAlternative = new Alternative(normVector);

            return getScoringFunction().computeScore(normAlternative, rule);
        }

        return 0;
    }

    public double getAlternativeScore(IAlternative alternative) {
        double[] unNormVector = alternative.getVector();
        double[] normVector = getNormalizer().normalize(unNormVector, getNormalizationTechnique(), false);
        IAlternative normAlternative = new Alternative(normVector);

        return getScoringFunction().computeScore(normAlternative, rule);

    }

    private DecisionRule updateRule(DecisionRule rule) {
        String[] antecedentItems = getDataset().getAntecedentItemsArray();
        String[] consequentItems = getDataset().getConsequentItemsArray();

        int[] antecedentShuffle = RandomUtil.randomShuffle(antecedentItems.length);
        int[] consequentShuffle = RandomUtil.randomShuffle(consequentItems.length);

        processAntecedents(rule, antecedentItems, antecedentShuffle);
        processConsequents(rule, consequentItems, consequentShuffle);

        return rule;
    }

    private void processAntecedents(DecisionRule rule, String[] antecedentItems, int[] antecedentShuffle) {
        for (int i = 0; i < antecedentShuffle.length; i++) {
            updateNormalization(rule);

            double originalScore = getValidRuleScore(rule);
            rule.addToX(antecedentItems[antecedentShuffle[i]]);
            double modifiedScore = getValidRuleScore(rule);

            if (isCertaintyHighEnough(modifiedScore, originalScore)) {
                break;
            }

            rule.removeFromX(antecedentItems[antecedentShuffle[i]]);
        }
    }

    private void processConsequents(DecisionRule rule, String[] consequentItems, int[] consequentShuffle) {
        for (int i = 0; i < consequentShuffle.length; i++) {
            updateNormalization(rule);

            double originalScore = getValidRuleScore(rule);
            String originalConsequent = rule.getY();
            rule.setY(consequentItems[consequentShuffle[i]]);
            double modifiedScore = getValidRuleScore(rule);

            if (isCertaintyHighEnough(modifiedScore, originalScore)) {
                break;
            }

            rule.setY(originalConsequent);
        }
    }

    private boolean isCertaintyHighEnough(double modifiedScore, double originalScore) {
        double certainty = modifiedScore == 0 ? 0 : getOutRankingCertainty().computeScore(modifiedScore, originalScore);
        return getRandom().Bernoulli(certainty);
    }

    private void updateNormalization(DecisionRule rule) {
        getNormalizer().normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);
    }

    private void initNormalization() {
        List<DecisionRule> validRules = getDataset().getRandomValidRules(100, smoothCounts, measureNames);

        for (DecisionRule rule : validRules)
            updateNormalization(rule);
    }

    @Override
    public void setScoringFunction(ISinglevariateFunction scoringFunction) {
        this.scoringFunction = scoringFunction;
        this.outRankingCertainty.setScoreFunction(scoringFunction);
    }
}
