package tools.functions.singlevariate.Choquet;

import java.util.BitSet;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;
import tools.utils.kappalab.NormalizedCapacity;

public class ChoquetScoreFunction implements ISinglevariateFunction {
    public static String TYPE = "normalizedChoquet";
    public @Setter @Getter String name = "normalizedChoquet";
    private @Getter @Setter IAlternative nadir, ideal;

    private NormalizedCapacity capacity;

    public ChoquetScoreFunction(NormalizedCapacity capacity) {
        this.capacity = capacity;
    }

    public ChoquetScoreFunction(int nbCriteria) {
        double equalWeight = 1.0 / (Math.pow(2, nbCriteria) - 2);
        this.capacity = new NormalizedCapacity(nbCriteria, equalWeight);
    }

    @Override
    public double computeScore(IAlternative alternative) {
        int nbCriteria = alternative.getVector().length;
        BitSet capacitySet = new BitSet(nbCriteria);
        capacitySet.set(0, nbCriteria);
        double score = 0d;
        double prevValue = 0d;
        for (int i = 0; i < nbCriteria; i++) {
            score += (alternative.getOrderedValue(i) - prevValue) * capacity.getCapacityValue(capacitySet);
            capacitySet.set(alternative.getOrderedPermutation()[i], false);
            prevValue = alternative.getOrderedValue(i);
        }
        return score;
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(rule);
    }
}