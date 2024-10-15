package tools.functions.singlevariate;

import java.util.Comparator;
import java.util.TreeSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.rules.DecisionRule;
import tools.utils.FunctionUtil;
import tools.utils.RuleUtil;

@Getter
@Setter
class AlternativeScore {
    private IAlternative alternative;
    private IAlternative normalized;
    private DecisionRule rule;
    private double score;

    public AlternativeScore(IAlternative alternative, IAlternative normalized, DecisionRule rule, double score) {
        this.alternative = alternative;
        this.normalized = normalized;
        this.score = score;
        this.rule = rule;
    }

    public AlternativeScore(AlternativeScore original) {
        this.alternative = original.alternative != null ? new Alternative((Alternative) original.getAlternative())
                : null;
        this.normalized = original.normalized != null ? new Alternative((Alternative) original.getNormalized()) : null;
        this.score = original.score;
        this.rule = original.rule != null ? RuleUtil.deepCopy(original.getRule()) : null;
    }
}

public class MinGapsScoreFunction implements ISinglevariateFunction {

    public static String TYPE = "minGaps";
    public @Setter @Getter String name = "minGaps";

    /** The function we want to compute the score for */
    private @Setter @Getter ISinglevariateFunction scoreFunction;

    /** List of already sampled alternatives and their associated scores */
    TreeSet<AlternativeScore> alreadySampled = new TreeSet<>(Comparator.comparingDouble(AlternativeScore::getScore));

    /** The current min gap value and the pair of alternatives that achieves it. */
    private @Getter @Setter IAlternative[] minGapPair;
    private @Getter @Setter DecisionRule[] rulesPair;
    private @Getter @Setter double minGap;

    /** The alternatives used for normalization */
    private @Getter @Setter IAlternative nadir, ideal;

    /** The interval of score in which we consider rules for min gap */
    private @Setter double[] scoreInterval;

    /** The tolerance used to derive the score interval */
    private double tolerance;

    /** Iterations */
    private @Setter @Getter boolean EXPLORATION;

    /**
     * Constructor for MinGapsScoreFunction.
     * 
     * @param scoreFunction The function used to compute scores for alternatives.
     */
    public MinGapsScoreFunction(ISinglevariateFunction scoreFunction, double tolerance) {
        this.scoreFunction = scoreFunction;
        this.minGap = Double.POSITIVE_INFINITY;
        this.tolerance = tolerance;
        this.EXPLORATION = true;
    }

    /**
     * This function computes the inverse minimum gap distance between the score of
     * the alternative t and the already sampled alternatives. If the gap is smaller
     * then the score will be higher.
     */
    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        // If we are still in the exploration phase we return a score of zero after
        // having updated the min and max
        if (this.EXPLORATION)
            return 0.0d;

        // After each iteration normalize all of the already seen alternatives with the
        // new values
        for (AlternativeScore node : alreadySampled) {
            IAlternative notNormalized = node.getAlternative();
            node.setNormalized(notNormalized);
            node.setScore(scoreFunction.computeScore(node.getNormalized(), node.getRule()));
        }

        if (getMinGapPair() != null) {
            // We recompute the min gap using the new normalization
            double scorePairFirst = scoreFunction.computeScore(getMinGapPair()[0], getRulesPair()[0]);
            double scorePairSecond = scoreFunction.computeScore(getMinGapPair()[1], getRulesPair()[1]);
            double newGap = Math.abs(scorePairFirst - scorePairSecond);

            // If the minGap changes due to normalization, we forget about the rule history
            // if(newGap != getMinGap())
            //     this.alreadySampled = new TreeSet<>(Comparator.comparingDouble(AlternativeScore::getScore));

            setMinGap(newGap);
        }

        IAlternative normalizedAlternative = alternative;

        // Compute the score for the given alternative
        double score = scoreFunction.computeScore(normalizedAlternative, rule);

        if (alreadySampled.isEmpty()) {
            AlternativeScore node = new AlternativeScore(new Alternative((Alternative) alternative),
                    new Alternative((Alternative) alternative), (DecisionRule) rule, score);
            alreadySampled.add(node);
            setScoreInterval(new double[]{score * (1d - tolerance), score * (1d + tolerance) }); 
            return 0.0d;
        }

        if (score < scoreInterval[0] || score > scoreInterval[1]) {
            double gap = Math.max(Math.abs(scoreInterval[0] - score),
                    Math.abs(Math.abs(scoreInterval[1] - score)));

            IAlternative nearest = alreadySampled.first().getAlternative();
            if (gap < this.minGap && alreadySampled.size() == 1 && !nearest.equals(alternative)) {
                setMinGap(gap);
                setMinGapPair(new IAlternative[] {  alreadySampled.first().getAlternative(), alternative });
                setRulesPair(new DecisionRule[] { alreadySampled.first().getRule(), rule });
            }

            return 1 - gap;
        }

        AlternativeScore floor = alreadySampled.floor(new AlternativeScore(null, null, null, score));
        AlternativeScore ceiling = alreadySampled.ceiling(new AlternativeScore(null, null, null, score));

        AlternativeScore nearest = null;
        if (floor == null) {
            nearest = ceiling;
        } else if (ceiling == null) {
            nearest = floor;
        } else {
            nearest = (score - floor.getScore() <= ceiling.getScore() - score) ? floor : ceiling;
        }

        double gap = Math.abs(score - nearest.getScore());
        double gapScore = 1.0 - gap;

        if (!nearest.getAlternative().equals(alternative)) {
            AlternativeScore node = new AlternativeScore(new Alternative((Alternative) alternative),
                    FunctionUtil.minMaxNormalize(new Alternative((Alternative) alternative), getIdeal(), getNadir()), (DecisionRule) rule, score);
            alreadySampled.add(node);

            if (gap < getMinGap()) {
                setMinGap(gap);
                setMinGapPair(new IAlternative[] { nearest.getAlternative(), alternative });
                setRulesPair(new DecisionRule[] { nearest.getRule(), rule });
            }

            return gapScore;
        }
        return 0.0d;
    }

    @Override
    public double computeScore(DecisionRule rule) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeScore'");
    }

    @Override
    public double computeScore(IAlternative alternative) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeScore'");
    }

}