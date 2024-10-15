package sampling;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.IMultivariateFunction;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.MinGapsScoreFunction;
import tools.functions.singlevariate.OWA.OWAScoreFunction;
import tools.rules.DecisionRule;
import tools.utils.AlternativeUtil;

public class MMASTest {
    private static Dataset dataset;
    private static ISinglevariateFunction scoringFunction;
    private static CertaintyFunction outRankingCertainty;
    private static PairwiseUncertainty pairwiseUncertainty;
    private static String[] measureNames;
    private static double smoothCounts;

    @BeforeAll
    static void setUp() throws IOException {
        // Mock dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("145");
        classItemValues.add("146");

        dataset = new Dataset("adult.dat", "src/test/resources/", classItemValues);

        // Measures
        measureNames = new String[] { "lift", "confidence", "support", "yuleQ", "kruskal" };
        smoothCounts = 1e-6d;

        double[] weights = new double[measureNames.length];
        Arrays.fill(weights, 1.0 / measureNames.length);
        scoringFunction = new OWAScoreFunction(weights);

        outRankingCertainty = new ScoreDifference(scoringFunction);

        pairwiseUncertainty = new PairwiseUncertainty("ScoreDifferencePairwise", outRankingCertainty);
    }

    @Test
    void testMMAS() throws IOException {
        int maxIterations = 1000;
        String outputDir = "src/test/output/";

        MMAS mmas = new MMAS(maxIterations, 1, dataset, pairwiseUncertainty, measureNames);

        // Run the MMAS algorithm
        DecisionRule[] resultRule = mmas.sample().get(0);

        // Construct filename based on the normalization method
        String filename = String.format("MMAS_adaptive_%d_%s.txt", maxIterations, outRankingCertainty.getName());

        // Print results to file
        printResultToFile(resultRule, outputDir, filename);
    }

    // @Test
    // void testMinGaps() throws IOException {
    // int maxIterations = 100;
    // String outputDir = "src/test/output/";

    // MinGapsScoreFunction minGaps = new MinGapsScoreFunction(scoringFunction,
    // 1000d);
    // SMAS smas = new SMAS(maxIterations, dataset, minGaps, measureNames, 1);

    // // Run the SMAS algorithm
    // smas.sample();

    // DecisionRule[] resultRule = minGaps.getRulesPair();

    // // Construct filename based on the normalization method
    // String filename = String.format("MinGaps_%d_%s.txt", maxIterations,
    // outRankingCertainty.getName());

    // // Print results to file
    // printResultToFile(resultRule, outputDir, filename);
    // }

    private void printResultToFile(DecisionRule[] rules, String outputDir, String filename) {
        StringBuilder sb = new StringBuilder();

        // Print rules information
        sb.append("Maximizing Rules:\n");
        sb.append("Antecedents: ").append(rules[0].getItemsInX()).append(" | ");
        sb.append("Consequent: ").append(rules[0].getY()).append("\n");
        sb.append("Antecedents: ").append(rules[1].getItemsInX()).append(" | ");
        sb.append("Consequent: ").append(rules[1].getY()).append("\n");

        // Compute alternative based on the rules
        IAlternative alternative0 = AlternativeUtil.computeAlternative(rules[0], dataset.getNbTransactions(),
                smoothCounts, measureNames);
        IAlternative alternative1 = AlternativeUtil.computeAlternative(rules[1], dataset.getNbTransactions(),
                smoothCounts, measureNames);

        sb.append("Alternative 1: ");
        for (double measure : alternative0.getVector()) {
            sb.append(String.format("%-10f", measure));
        }

        sb.append("\n");

        sb.append("Alternative 0: ");
        for (double measure : alternative1.getVector()) {
            sb.append(String.format("%-10f", measure));
        }

        // Calculate score
        double score = pairwiseUncertainty.computeScore(rules);
        sb.append("\nScore: ").append(1 - score).append("\n");

        // Write to file
        try (FileWriter writer = new FileWriter(outputDir + filename)) {
            writer.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
