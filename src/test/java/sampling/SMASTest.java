package sampling;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.functions.multivariate.outRankingCertainties.Thurstone;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.OWA.OWAScoreFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;
import tools.utils.AlternativeUtil;
import tools.utils.RuleUtil;

class SMASTest {

    private static Dataset dataset;
    private static ISinglevariateFunction scoringFunction;
    private static CertaintyFunction outRankingCertainty;
    private static String[] measureNames;
    private static double smoothCounts;

    @BeforeAll
    static void setUp() throws IOException {
        // Mock dataset
        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("911");
        classItemValues.add("912");

        dataset = new Dataset("toms.dat", "src/test/resources/", classItemValues);

        // Measures
        measureNames = new String[] { "lift", "confidence", "support", "yuleQ", "kruskal" };
        smoothCounts = 1e-6;

        double[] weights = new double[measureNames.length];
        Arrays.fill(weights, 1.0 / measureNames.length);
        scoringFunction = new OWAScoreFunction(weights);
    }

    @Test
    void testSMASRunsCertainty() throws IOException {
        int maxIterations = 10;
        String outputDir = "src/test/output/";

        List<SMAS> samplerList = new ArrayList<>();
        // Create SMAS instance with specific normalization method
        SMAS smasSD = new SMAS(maxIterations, dataset, new ScoreDifference(scoringFunction), scoringFunction,
                measureNames,
                smoothCounts, 1);
        SMAS smasBT = new SMAS(maxIterations, dataset, new BradleyTerry(scoringFunction), scoringFunction, measureNames,
                smoothCounts, 1);
        SMAS smasTh = new SMAS(maxIterations, dataset, new Thurstone(scoringFunction), scoringFunction, measureNames,
                smoothCounts, 1);

        samplerList.add(smasSD);
        samplerList.add(smasBT);
        samplerList.add(smasTh);

        for (SMAS smas : samplerList) {
            smas.setScoringFunction(scoringFunction);
            smas.setNormalizationTechnique(NormalizationMethod.NO_NORMALIZATION);

            // Run the SMAS algorithm
            DecisionRule resultRule = smas.sample().get(0);

            // Assert that the rule is not null
            assertNotNull(resultRule,
                    "Decision rule should not be null for method: " + smas.getOutRankingCertainty().getName());

            // Construct filename based on the normalization method
            String filename = String.format("SMAS_%d_%s.txt", maxIterations, smas.getOutRankingCertainty().getName());

            // Print results to file
            printResultToFile(resultRule, outputDir, filename, smas);
        }
    }

    // @Test
    // void testUnrestrictedSampler() throws IOException {
    // int maxIterations = 100;

    // UnrestrictedSampler sampler = new UnrestrictedSampler(dataset,
    // maxIterations);

    // List<DecisionRule> sample = sampler.sample();
    // DecisionRule resultRule = sample.stream()
    // .max((rule1, rule2) -> {
    // double score1 = getValidRuleScore(rule1);
    // double score2 = getValidRuleScore(rule2);
    // return Double.compare(score1, score2);
    // })
    // .orElse(null);

    // System.out.println("Best score found: " + getValidRuleScore(resultRule));
    // }

    public double getValidRuleScore(DecisionRule rule) {
        if (RuleUtil.isValid(rule)) {
            IAlternative alt = AlternativeUtil.computeAlternativeOrZero(rule, dataset.getNbTransactions(),
                    smoothCounts, measureNames);

            return scoringFunction.computeScore(alt, rule);
        }

        return 0;
    }

    void testSMASRuns() throws IOException {
        int maxIterations = 100;
        String outputDir = "src/test/output/";

        // Iterate over each normalization method
        Arrays.stream(NormalizationMethod.values()).parallel().forEach(method -> {
            // Create SMAS instance with specific normalization method
            SMAS smas = new SMAS(maxIterations, dataset, outRankingCertainty, scoringFunction, measureNames,
                    smoothCounts, 1);
            smas.setScoringFunction(scoringFunction);
            smas.setNormalizationTechnique(method);

            // Run the SMAS algorithm
            DecisionRule resultRule = smas.sample().get(0);

            // Assert that the rule is not null
            assertNotNull(resultRule, "Decision rule should not be null for method: " + method);

            // Construct filename based on the normalization method
            String filename = String.format("SMAS_%d_%s_%s.txt", maxIterations, method.name(),
                    outRankingCertainty.getClass().getSimpleName());

            // Print results to file
            printResultToFile(resultRule, outputDir, filename, smas);
        });
    }

    void testSMASRPerformance() throws IOException {
        int maxIterations = 100;
        String outputDir = "src/test/output/";

        // List of out ranking certainties to test
        CertaintyFunction[] outRankingCertainties = {
                new Thurstone(scoringFunction),
                new BradleyTerry(scoringFunction),
                new ScoreDifference(scoringFunction)
        };

        // Table to store the results
        double[][] means = new double[outRankingCertainties.length][NormalizationMethod.values().length];
        double[][] variances = new double[outRankingCertainties.length][NormalizationMethod.values().length];

        // Iterate over each out ranking certainty
        for (int i = 0; i < outRankingCertainties.length; i++) {
            final int certaintyIndex = i;

            // Iterate over each normalization method in parallel
            Arrays.stream(NormalizationMethod.values()).parallel().forEach(method -> {
                List<Double> scores = new ArrayList<>();

                IntStream.range(0, 10).parallel().forEach(run -> {
                    // Create SMAS instance with specific normalization method
                    SMAS smas = createSMAS(maxIterations, dataset, outRankingCertainties[certaintyIndex],
                            scoringFunction, measureNames, smoothCounts, method);

                    // Run the SMAS algorithm
                    DecisionRule resultRule = smas.sample().get(0);
                    if (resultRule == null) {
                        throw new IllegalStateException("Decision rule should not be null for method: " + method);
                    }

                    // Calculate score and add to the list
                    double score = smas.getScoringFunction().computeScore(resultRule.getAlternative());
                    synchronized (scores) {
                        scores.add(score);
                    }
                });
                // Calculate mean and variance for the scores
                double mean = scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                double variance = scores.stream().mapToDouble(score -> Math.pow(score - mean, 2)).average().orElse(0.0);

                // Store results in the tables
                int methodIndex = method.ordinal();
                means[certaintyIndex][methodIndex] = mean;
                variances[certaintyIndex][methodIndex] = variance;
            });
        }

        // Generate the markdown table
        generateMarkdownTable(means, variances, outRankingCertainties, NormalizationMethod.values(), outputDir);
    }

    private static SMAS createSMAS(int maxIterations, Dataset dataset, CertaintyFunction outRankingCertainty,
            ISinglevariateFunction scoringFunction, String[] measureNames, double smoothCounts,
            NormalizationMethod method) {

        // Create SMAS instance with specific normalization method
        SMAS smas = new SMAS(maxIterations, dataset, outRankingCertainty, scoringFunction,
                measureNames, smoothCounts, 1);
        smas.setScoringFunction(scoringFunction);
        smas.setNormalizationTechnique(method);

        return smas;
    }

    private void generateMarkdownTable(double[][] means, double[][] variances,
            CertaintyFunction[] outRankingCertainties,
            NormalizationMethod[] normalizationMethods, String outputDir) throws IOException {
        StringBuilder sb = new StringBuilder();

        String pm = "\u00B1"; // ± symbol

        // Create header
        sb.append("| Out Ranking Certainty / Normalization Method | ");
        sb.append(Arrays.stream(normalizationMethods)
                .map(Enum::name)
                .collect(Collectors.joining(" | ")));
        sb.append(" |\n");

        // Create separator line
        sb.append("|---------------------------------------------|");
        for (int i = 0; i < normalizationMethods.length; i++) {
            sb.append("------------|"); // Manually appending the separator for each column
        }
        sb.append("\n");

        // Find the maximum score for each normalization method (column)
        double[] maxScores = new double[normalizationMethods.length];
        for (int j = 0; j < normalizationMethods.length; j++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < outRankingCertainties.length; i++) {
                if (means[i][j] > max) {
                    max = means[i][j];
                }
            }
            maxScores[j] = max;
        }

        // Add rows
        for (int i = 0; i < outRankingCertainties.length; i++) {
            sb.append("| ").append(outRankingCertainties[i].getClass().getSimpleName()).append(" | ");
            for (int j = 0; j < normalizationMethods.length; j++) {
                if (means[i][j] == maxScores[j]) {
                    // Wrap the best score in bold
                    sb.append(String.format("**%.4f** %s %.4f | ", means[i][j], pm, variances[i][j]));
                } else {
                    sb.append(String.format("%.4f %s %.4f | ", means[i][j], pm, variances[i][j]));
                }
            }
            sb.append("\n");
        }

        // Write the table to a file
        try (FileWriter writer = new FileWriter(outputDir + "SMASTestResults.md")) {
            writer.write(sb.toString());
        }
    }

    private void printResultToFile(DecisionRule rule, String outputDir, String filename, SMAS smas) {
        StringBuilder sb = new StringBuilder();

        // Print rule information
        sb.append("Maximizing Rule:\n");
        sb.append("Antecedents: ").append(rule.getItemsInX()).append("\n");
        sb.append("Consequent: ").append(rule.getY()).append("\n");

        // Compute alternative based on the rule
        IAlternative alternative = AlternativeUtil.computeAlternative(rule, dataset.getNbTransactions(),
                smoothCounts, measureNames);

        sb.append("Measures: ");
        for (double measure : alternative.getVector()) {
            sb.append(String.format("%-10f", measure));
        }

        // Calculate score
        double score = scoringFunction.computeScore(rule.getAlternative());
        sb.append("\nScore: ").append(score).append("\n");

        // Add cumulative distribution histogram of score history
        sb.append("\nCumulative Distribution of Score History (10 bins):\n");

        // Retrieve the score history
        List<Double> scoreHist = smas.getScoreHistory();

        // Define 10 bins
        int numBins = 10;
        double minScore = scoreHist.stream().min(Double::compareTo).orElse(0.0);
        double maxScore = scoreHist.stream().max(Double::compareTo).orElse(1.0);
        double binSize = (maxScore - minScore) / numBins;

        // Array to store counts per bin
        int[] bins = new int[numBins];

        // Count how many scores fall into each bin
        for (double s : scoreHist) {
            int binIndex = (int) Math.min((s - minScore) / binSize, numBins - 1);
            bins[binIndex]++;
        }

        // Calculate cumulative counts and print histogram
        int cumulativeCount = 0;
        for (int i = 0; i < numBins; i++) {
            cumulativeCount += bins[i];
            double binRangeStart = minScore + i * binSize;
            double binRangeEnd = binRangeStart + binSize;
            sb.append(String.format("Bin %2d [%10.4f - %10.4f]: %d (Cumulative: %d)\n", i + 1, binRangeStart,
                    binRangeEnd, bins[i], cumulativeCount));
        }

        // Write to file
        try (FileWriter writer = new FileWriter(outputDir + filename)) {
            writer.write(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
