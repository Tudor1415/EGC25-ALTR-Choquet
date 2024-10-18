package sampling;

import java.util.ArrayList;
import java.util.List;

import tools.data.Dataset;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;
import tools.utils.RuleUtil;

public class UnrestrictedSampler implements Sampler {
    private Dataset dataset;
    private int numIterations;

    public UnrestrictedSampler(Dataset dataset, int numIterations) {
        this.dataset = dataset;
        this.numIterations = numIterations;
    }

    public List<DecisionRule> sample() {
        // Initialize with a random rule
        DecisionRule currentRule = dataset.getRandomValidRules(1, 1.0, new String[] { "support", "confidence" }).get(0);
        List<DecisionRule> sample = new ArrayList<>();

        for (int iter = 0; iter < numIterations; iter++) {
            String[] conseauentsArray = dataset.getConsequentItemsArray();

            for (int i = 0; i < conseauentsArray.length; i++) {
                String consequentValue = conseauentsArray[i];

                double probability = getConditionalProb(currentRule, consequentValue, "consequent", 1000);

                if (new RandomUtil().Bernoulli(probability)) {
                    currentRule.setY(consequentValue);
                }
            }

            String[] antecedentsArray = dataset.getAntecedentItemsArray();

            for (int i = 0; i < antecedentsArray.length; i++) {
                String antecedentValue = antecedentsArray[i];

                double probability = getConditionalProb(currentRule, antecedentValue, "antecedent", 1000);

                if (new RandomUtil().Bernoulli(probability)) {
                    currentRule.addToX(antecedentValue);
                }
            }
            if (RuleUtil.isValid(currentRule))
                sample.add(RuleUtil.deepCopy(currentRule));
        }

        return sample;
    }

    private double g(DecisionRule rule) {
        double support = rule.getFreqX() / (double) dataset.getNbTransactions();
        double confidence = rule.getFreqY() / (double) rule.getFreqX();
        return support * confidence;
    }

    /**
     * Computes the conditional probability P(J_s = 1 | J_-s).
     * 
     * @param J         The rule that will be conditioned.
     * @param itemValue The value of the item that is to be added or not.
     * @param type      Indicates if the item is a consequent or antecedent
     *                  item.
     * @param csi       The value of csi.
     * @return The probability that the conditioned item is present.
     */
    public double getConditionalProb(DecisionRule J, String itemValue, String type, double csi) {
        // Keeping the last class in memory
        String lastClass = J.getY();

        // Returning the conditional probability P(J_s = 0 | J_-s)
        double without = Math.exp(csi * g(J));

        // Adding the item to the antecedent or consequent of the rule
        RuleUtil.addItemToRule(J, itemValue, type);

        // Returning the conditional probability P(J_s = 1 | J_-s)
        double with = Math.exp(csi * g(J));

        // Computing the probability P(J_s = 1)
        double probability = with / (with + without);

        // Removing the added item from the rule
        RuleUtil.removeItemFromRule(J, itemValue, type, lastClass);

        return probability;
    }
}
