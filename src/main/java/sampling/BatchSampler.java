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

public class BatchSampler extends SMAS {

    public BatchSampler(int maximumIterations, Dataset dataset, ISinglevariateFunction scoringFunction,
            String[] measureNames, int topK) {
        super(maximumIterations, dataset, scoringFunction, measureNames, topK);
    }

    @Override
    protected void processAntecedents(DecisionRule rule, String[] antecedentItems, int[] antecedentShuffle) {

        // Random skip to half the rule
        skipToHalf(rule);

        for (int i = 0; i < antecedentShuffle.length; ++i) {
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

    private void skipToHalf(DecisionRule rule) {
        Set<String> halfAntecedent = splitSet(rule.getItemsInX()).get(0);

        double originalScore = getValidRuleScore(rule);

        for (String item : halfAntecedent)
            rule.removeFromX(item);

        double modifiedScore = getValidRuleScore(rule);

        if (!isCertaintyHighEnough(modifiedScore, originalScore)) {
            for (String item : halfAntecedent)
                rule.addToX(item);
        }
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
