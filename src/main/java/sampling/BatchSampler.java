package sampling;

import tools.data.Dataset;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;

public class BatchSampler extends SMAS {

    public BatchSampler(int maximumIterations, Dataset dataset, ISinglevariateFunction scoringFunction,
            String[] measureNames, int topK) {
        super(maximumIterations, dataset, scoringFunction, measureNames, topK);
    }

    @Override
    protected void processAntecedents(DecisionRule rule, String[] antecedentItems, int[] antecedentShuffle) {
        int antSize = rule.getItemsInX().size();
        int step = Math.min(Math.floorDiv(antSize, 2), 0);

        for (int i = 0; i < antecedentShuffle.length; i += step) {
            updateNormalization(rule);

            double originalScore = getValidRuleScore(rule);

            // Add items in batch of size step
            for (int j = i; j < i + step; ++j)
                rule.addToX(antecedentItems[antecedentShuffle[i]]);

            double modifiedScore = getValidRuleScore(rule);

            if (isCertaintyHighEnough(modifiedScore, originalScore)) {
                break;
            }

            rule.removeFromX(antecedentItems[antecedentShuffle[i]]);
        }
    }
}
