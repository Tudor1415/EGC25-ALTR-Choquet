package tools.oracles;

import tools.rules.DecisionRule;

public interface Oracle {

    /**
     * Compares two rules based on human preference.
     *
     * @param rule_a The first rule to compare.
     * @param rule_b The second rule to compare.
     * @return A negative integer, zero, or a positive integer if the score of
     *         rule 'a' is less than, equal to, or greater than the score of
     *         rule 'b', respectively.
     */
    int compare(DecisionRule rule_a, DecisionRule rule_b);
}