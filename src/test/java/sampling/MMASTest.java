package sampling;

import java.util.Set;
import java.util.Arrays;
import java.util.HashSet;
import java.io.FileWriter;
import java.io.IOException;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;

import tools.data.Dataset;
import tools.rules.DecisionRule;
import tools.utils.AlternativeUtil;
import tools.alternatives.IAlternative;
import tools.utils.kappalab.NormalizedCapacity;
import tools.functions.multivariate.RandomDomination;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.PairwiseUncertainty;
import tools.functions.multivariate.IMultivariateFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.Choquet.ChoquetScoreFunction;
import tools.functions.multivariate.outRankingCertainties.Sensitivity;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;

public class MMASTest {
    private static Dataset dataset;
    private static ISinglevariateFunction scoringFunction;
    private static CertaintyFunction outRankingCertainty;
    private static PairwiseUncertainty pairwiseUncertainty;
    private static CertaintyFunction pairwiseSensitivity;
    private static String[] measureNames;
    private static double smoothCounts;

    @BeforeAll
    static void setUp() throws IOException {
        // Mock dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("346");
        classItemValues.add("347");

        dataset = new Dataset("dota.dat", "src/test/resources/", classItemValues);

        // Measures
        measureNames = new String[] { "lift", "confidence", "support", "yuleQ", "kruskal" };
        smoothCounts = 1e-6d;

        double[] weights = new double[measureNames.length];
        Arrays.fill(weights, 1.0 / measureNames.length);

        NormalizedCapacity randomCapacity = new NormalizedCapacity(measureNames.length, true);
        scoringFunction = new ChoquetScoreFunction(randomCapacity);

        outRankingCertainty = new BradleyTerry(scoringFunction);
        IMultivariateFunction regularization = new RandomDomination(measureNames.length);

        pairwiseUncertainty = new PairwiseUncertainty("BradleyTerryPairUncertainty", outRankingCertainty,
                regularization);
        pairwiseSensitivity = new Sensitivity("Sensitivity", scoringFunction);
    }

    @Test
    void testMMAS() throws IOException {
        int maxIterations = 100;
        String outputDir = "src/test/output/";

        MMAS mmas = new MMAS(maxIterations, 1, dataset, pairwiseUncertainty, measureNames);
        mmas.setUncertainty(false);
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

        // Calculate and print scores
        IMultivariateFunction uncertainty = new PairwiseUncertainty("Uncertainty", outRankingCertainty);
        double uncertaintyScore = uncertainty.computeScore(rules);
        double regularizationScore = ((IMultivariateFunction) pairwiseUncertainty.getRegularization())
                .computeScore(rules);
        double combinedScore = pairwiseUncertainty.computeScore(rules);

        sb.append("\nUncertainty Score: ").append(uncertaintyScore).append("\n");
        sb.append("Regularization Score: ").append(regularizationScore).append("\n");
        sb.append("Combined Score: ").append(combinedScore).append("\n");

        // Write to file
        try (FileWriter writer = new FileWriter(outputDir + filename)) {
            writer.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
