package sampling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tools.data.Dataset;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.rules.DecisionRule;
import tools.utils.RuleUtil;

public class BatchSampler extends SMAS {

    public BatchSampler(int maximumIterations, Dataset dataset, ISinglevariateFunction scoringFunction,
            String[] measureNames, int topK) {
        super(maximumIterations, dataset, scoringFunction, measureNames, topK);
    }

    @Override
    protected void processAntecedents(DecisionRule rule, String[] antecedentItems, int[] antecedentShuffle) {
        DecisionRule bestRule = RuleUtil.simpleCopy(rule);
        double bestScore = getValidRuleScore(bestRule);

        skipToHalf(rule);

        for (int i = 0; i < antecedentShuffle.length; i++) {
            updateNormalization(rule);

            double originalScore = getValidRuleScore(rule);
            rule.addToX(antecedentItems[antecedentShuffle[i]]);
            double modifiedScore = getValidRuleScore(rule);

            if (!isCertaintyHighEnough(modifiedScore, originalScore)) {
                rule.removeFromX(antecedentItems[antecedentShuffle[i]]);
            } else if (modifiedScore > bestScore) {
                bestRule = RuleUtil.simpleCopy(rule);
                bestScore = modifiedScore;
            }
        }

        rule = bestRule;
    }

    private void skipToHalf(DecisionRule rule) {
        Set<String> halfAntecedent = splitSet(rule.getItemsInX()).get(0);

        double originalScore = getValidRuleScore(rule);

        rule.removeFromX(halfAntecedent);

        double modifiedScore = getValidRuleScore(rule);

        if (!isCertaintyHighEnough(modifiedScore, originalScore)) 
            rule.addToX(halfAntecedent);
    }

    public static <T> List<Set<T>> splitSet(Set<T> originalSet) {
        // Convert the set to a list
        List<T> list = new ArrayList<>(originalSet);

        // Shuffle the list to randomize the order
        Collections.shuffle(list);

        // Calculate the size of each subset
        int size = list.size() / 2;

        // Create two sets for the split
        Set<T> set1 = new HashSet<>(list.subList(0, size));
        Set<T> set2 = new HashSet<>(list.subList(size, list.size()));

        // Return a list containing the two sets
        return Arrays.asList(set1, set2);
    }
}
