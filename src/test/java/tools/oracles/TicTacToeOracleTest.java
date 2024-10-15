package tools.oracles;

import static tools.rules.RuleMeasures.addedValue;
import static tools.rules.RuleMeasures.certainty;
import static tools.rules.RuleMeasures.cosine;
import static tools.rules.RuleMeasures.kruskal;
import static tools.rules.RuleMeasures.yuleQ;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

import tools.data.Dataset;
import tools.rules.DecisionRule;

public class TicTacToeOracleTest {
    @Test
    void testShowVisualizationForClass28And29() throws IOException {
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("28");
        classItemValues.add("29");

        String[] measureNames = { yuleQ, cosine, kruskal, addedValue, certainty };

        Dataset dataset = new Dataset("tictactoe.dat", "src/test/resources/", classItemValues);

        // Define antecedents and consequents for rule_a and rule_b
        Set<String> itemsInX_a = new HashSet<>(Arrays.asList("1", "5", "9")); // b, o, x
        Set<String> itemsInX_b = new HashSet<>(Arrays.asList("3", "6", "7")); // x, x, b
        String consequent_a = "28"; // Class 28 (green)
        String consequent_b = "29"; // Class 29 (red)

        // Create DecisionRule for rule_a and rule_b
        DecisionRule rule_a = new DecisionRule(itemsInX_a, consequent_a, dataset, 3, 3, 0.1, measureNames);
        DecisionRule rule_b = new DecisionRule(itemsInX_b, consequent_b, dataset, 3, 3, 0.1, measureNames);

        // Invoke the visualization method to see output on screen
        new TicTacToeOracle().showVisualization(rule_a, rule_b);
    }
}

// Rule A        |     Rule B
// b | o | x     |     b | o | x
// b | o | x     |     b | o | x
// b | o | x     |     b | o | x
