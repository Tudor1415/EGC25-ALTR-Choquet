package experiments;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import sampling.BatchSampler;
import sampling.SMAS;
import sampling.UnrestrictedSampler;
import sampling.Sampler;
import tools.alternatives.IAlternative;
import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.multivariate.outRankingCertainties.BradleyTerry;
import tools.functions.multivariate.outRankingCertainties.ScoreDifference;
import tools.functions.multivariate.outRankingCertainties.Thurstone;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.OWA.OWALexmin;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;
import tools.utils.AlternativeUtil;
import tools.utils.RuleUtil;

public class SamplingMethodExperiment {

    private static List<String> datasetNames = Arrays.asList("toms", "bank", "credit", "connect", "mushroom");
    private static String[] allMeasureNames = new String[] { "lift", "confidence", "support", "yuleQ", "kruskal",
            "cosine", "phi", "pavillon", "certainty" };

    private static String dataDirectory = "data/folds/";

    private static ISinglevariateFunction owa_score_function = new OWALexmin(0.01, allMeasureNames.length);

    /**
     * Creates the out-ranking certainties using the provided scoring function.
     */
    private static CertaintyFunction[] createOutRankingCertainties(ISinglevariateFunction scoringFunction) {

        return new CertaintyFunction[] {
                new Thurstone(scoringFunction),
                new BradleyTerry(scoringFunction),
                new ScoreDifference(scoringFunction)
        };
    }

    /**
     * Creates an SMAS instance for the experiment.
     */
    private static SMAS createSMAS(int maxIterations, Dataset dataset, CertaintyFunction outRankingCertainty,
            ISinglevariateFunction scoringFunction, String[] measureNames, double smoothCounts) {
        SMAS smas = new SMAS(maxIterations, dataset, outRankingCertainty, scoringFunction, measureNames, smoothCounts,
                maxIterations);
        smas.setScoringFunction(scoringFunction);
        smas.setNormalizationTechnique(NormalizationMethod.NO_NORMALIZATION);
        return smas;
    }

    private static BatchSampler createBatchSampler(int maxIterations, Dataset dataset,
            CertaintyFunction outRankingCertainty,
            ISinglevariateFunction scoringFunction, String[] measureNames, double smoothCounts) {
        BatchSampler BatchSampler = new BatchSampler(maxIterations, dataset, scoringFunction, measureNames,
                maxIterations);
        BatchSampler.setScoringFunction(scoringFunction);
        BatchSampler.setNormalizationTechnique(NormalizationMethod.NO_NORMALIZATION);
        return BatchSampler;
    }

    private static String ruleToString(DecisionRule rule) {
        Set<String> antecedentValues = rule.getItemsInX();
        String consequentValues = rule.getY();

        return "[" + String.join("; ", antecedentValues) + "]" + " => " + "[" + String.join("; ", consequentValues)
                + "]";
    }

    public static double getValidRuleScore(DecisionRule rule, Dataset dataset, double smoothCounts,
            String[] measureNames, ISinglevariateFunction scoringFunction) {
        if (RuleUtil.isValid(rule)) {
            IAlternative alt = AlternativeUtil.computeAlternativeOrZero(rule, dataset.getNbTransactions(),
                    smoothCounts, measureNames);

            return scoringFunction.computeScore(alt, rule);
        }

        return 0;
    }

    public static void writeSampleToCSV(List<DecisionRule> rules, List<Double> scoresApprox, String fileName,
            int samplingIterations, String outputDirectory) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String directoryPath = outputDirectory + samplingIterations + "/";
        String filePath = directoryPath + fileName + "_" + timestamp + ".csv";

        try {
            // Ensure the directory exists or create it
            Files.createDirectories(Paths.get(directoryPath));

            try (FileWriter writer = new FileWriter(filePath)) {

                // If rules is empty, we only write the header
                if (!rules.isEmpty()) {
                    String[] measureNames = rules.get(0).getMeasureNames(); // Get the measure names dynamically
                    writer.append("Rule,"); // Start with "Rule,"
                    for (String measureName : measureNames) {
                        writer.append(measureName).append(","); // Append each measure name with a comma
                    }
                    writer.append("scoreApprox,\n"); // End with "scoreApprox"
                } else {
                    // If no rules are available, write a default header
                    writer.append("Rule,confidence,support,scoreApprox,\n");
                }

                if (rules.isEmpty()) {
                    System.out.println("No rules available, writing empty CSV with header only.");
                    return; // Exit after writing the header
                }

                // Write data for each rule if available
                for (int i = 0; i < rules.size(); i++) {
                    writer.append(ruleToString(rules.get(i)) + ",");
                    double[] vector = rules.get(i).getAlternative().getVector();
                    for (double value : vector) {
                        writer.append(Double.toString(value)).append(",");
                    }
                    writer.append(scoresApprox.get(i) + ",");
                    writer.append("\n");
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Runs the experiment on a single fold of the dataset.
     */
    private static void runOnFold(Dataset dataset, ISinglevariateFunction scoreFunction, String[] measureNames,
            String datasetName, int foldIdx, int samplingIterations, String outputDirectory) {

        // Process regular SMAS sampling
        processSamplingForCertainties(dataset, scoreFunction, measureNames, datasetName, foldIdx, samplingIterations,
                outputDirectory);

        // Process Unrestricted Sampler
        // processUnrestrictedSampling(dataset, scoreFunction, datasetName, foldIdx,
        // samplingIterations, outputDirectory,
        // measureNames);

        // Process Batch Sampler
        processBatchSamplingForCertainties(dataset, scoreFunction, measureNames, datasetName, foldIdx,
                samplingIterations,
                outputDirectory);
    }

    private static void processSamplingForCertainties(Dataset dataset, ISinglevariateFunction scoreFunction,
            String[] measureNames, String datasetName, int foldIdx, int samplingIterations, String outputDirectory) {

        CertaintyFunction[] outRankingCertainties = createOutRankingCertainties(scoreFunction);

        for (CertaintyFunction certaintyFunction : outRankingCertainties) {
            SMAS sampler = createSMAS(samplingIterations, dataset, certaintyFunction, scoreFunction, measureNames,
                    0.01d);

            // Run the sampling with a timeout
            List<DecisionRule> sample = executeSamplingWithTimeout(sampler, 30);

            // Process the results after sampling
            String filename = createFileName(datasetName, foldIdx, scoreFunction, sampler, certaintyFunction);
            List<Double> approxScores = computeApproxScores(sample, scoreFunction);

            // Write the results to CSV
            writeSampleToCSV(sample, approxScores, filename, samplingIterations, outputDirectory);
        }
    }

    private static void processUnrestrictedSampling(Dataset dataset, ISinglevariateFunction scoreFunction,
            String datasetName, int foldIdx, int samplingIterations, String outputDirectory, String[] measureNames) {

        UnrestrictedSampler unrestrictedSampler = new UnrestrictedSampler(dataset, samplingIterations);

        // Run the unrestricted sampling with a timeout
        List<DecisionRule> unrestrictedSample = executeSamplingWithTimeout(unrestrictedSampler, 30);

        // Process the results after unrestricted sampling
        String filename = datasetName + "_" + foldIdx + "_" + scoreFunction.getName() + "_"
                + samplingIterations + "_UnrestrictedSampling";

        List<Double> approxScores = computeValidRuleScores(unrestrictedSample, dataset, measureNames, scoreFunction);

        // Write the results to CSV
        writeSampleToCSV(unrestrictedSample, approxScores, filename, samplingIterations, outputDirectory);
    }

    private static void processBatchSamplingForCertainties(Dataset dataset, ISinglevariateFunction scoreFunction,
            String[] measureNames, String datasetName, int foldIdx, int samplingIterations, String outputDirectory) {

        CertaintyFunction[] outRankingCertainties = createOutRankingCertainties(scoreFunction);

        // for (CertaintyFunction certaintyFunction : outRankingCertainties) {
        BatchSampler sampler = createBatchSampler(samplingIterations, dataset, null, scoreFunction, measureNames,
                0.01d);

        // Run the sampling with a timeout
        List<DecisionRule> sample = executeSamplingWithTimeout(sampler, 30);

        // Process the results after sampling
        String filename = datasetName + "_" + foldIdx + "_" + scoreFunction.getName() + "_"
                + samplingIterations + "_BatchSampling";
        List<Double> approxScores = computeApproxScores(sample, scoreFunction);

        // Write the results to CSV
        writeSampleToCSV(sample, approxScores, filename, samplingIterations, outputDirectory);
        // }
    }

    private static List<DecisionRule> executeSamplingWithTimeout(Sampler sampler, int timeoutInMinutes) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<List<DecisionRule>> future = executor.submit(() -> sampler.sample());

        List<DecisionRule> sample = new ArrayList<>();
        try {
            sample = future.get(timeoutInMinutes, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            System.err.println("Sampling timed out after " + timeoutInMinutes + " minutes.");
            future.cancel(true);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdownNow();
        }
        return sample;
    }

    private static String createFileName(String datasetName, int foldIdx, ISinglevariateFunction scoreFunction,
            SMAS sampler, CertaintyFunction certaintyFunction) {
        return String.format("%s_%d_%s_%d_%s",
                datasetName,
                foldIdx,
                scoreFunction.getName(),
                sampler.getMaximumIterations(),
                certaintyFunction.getName());
    }

    private static List<Double> computeApproxScores(List<DecisionRule> sample, ISinglevariateFunction scoreFunction) {
        return sample.parallelStream()
                .map(rule -> scoreFunction.computeScore(rule.getAlternative()))
                .collect(Collectors.toList());
    }

    private static List<Double> computeValidRuleScores(List<DecisionRule> sample, Dataset dataset,
            String[] measureNames, ISinglevariateFunction scoreFunction) {
        return sample.isEmpty()
                ? new ArrayList<>()
                : sample.parallelStream()
                        .map(rule -> getValidRuleScore(rule, dataset, 0.01d, measureNames, scoreFunction))
                        .collect(Collectors.toList());
    }

    private static Set<String> getClassItems(String datasetName) {
        switch (datasetName) {
            case "adult":
                return new HashSet<>(Arrays.asList("145", "146"));
            case "bank":
                return new HashSet<>(Arrays.asList("89", "90"));
            case "connect":
                return new HashSet<>(Arrays.asList("127", "128"));
            case "credit":
                return new HashSet<>(Arrays.asList("111", "112"));
            case "dota":
                return new HashSet<>(Arrays.asList("346", "347"));
            case "toms":
                return new HashSet<>(Arrays.asList("911", "912"));
            case "mushroom":
                return new HashSet<>(Arrays.asList("116", "117"));
            default:
                return null;
        }
    }

    public static List<Dataset> readDatasetsFromFold(String datasetName, String trainOrTest) throws IOException {
        String folderPath = dataDirectory + datasetName + trainOrTest;
        File folder = new File(folderPath);

        List<Dataset> datasets = new ArrayList<>();

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();

            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        Dataset dataset = new Dataset(file.getName(), folderPath, getClassItems(datasetName));
                        datasets.add(dataset);
                    }
                }
            } else {
                System.out.println("The folder is empty or cannot be read.");
            }
        } else {
            System.out.println("The specified path is not a valid directory.");
        }

        return datasets;
    }

    public static List<Dataset> readDatasets(String datasetName, String folderPath) throws IOException {
        File folder = new File(folderPath);

        List<Dataset> datasets = new ArrayList<>();

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();

            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        Dataset dataset = new Dataset(file.getName(), folderPath, getClassItems(datasetName));
                        datasets.add(dataset);
                    }
                }
            } else {
                System.out.println("The folder is empty or cannot be read.");
            }
        } else {
            System.out.println("The specified path is not a valid directory.");
        }

        return datasets;
    }

    public static void runOnFolds() {
        ExecutorService executor = Executors.newFixedThreadPool(10);
        try {
            for (int samplingIterations = 1; samplingIterations < 100_001; samplingIterations *= 10) {
                for (String datasetName : datasetNames) {
                    final int samplingIterationsFinal = samplingIterations;
                    System.out.println("Dataset: " + datasetName + " - Iterations: " + samplingIterations);
                    try {
                        List<Dataset> testDatasets = readDatasetsFromFold(datasetName, "/test/");

                        IntStream.range(0, testDatasets.size()).parallel().forEach(foldIdx -> {
                            Dataset testDataset = testDatasets.get(foldIdx);

                            try {
                                runOnFold(testDataset, owa_score_function, allMeasureNames, datasetName, foldIdx,
                                        samplingIterationsFinal, "sampling_experiment/samples/ratio_1to10/");
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                ;
            }

            executor.shutdown();
            executor.awaitTermination(24, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace(); // Log interruption exceptions
        }
    }

    public static void runOnDataset() throws IOException {
        // Process dataset names in parallel
        datasetNames.parallelStream().forEach(datasetName -> {
            try {
                Dataset dataset = new Dataset(datasetName + ".dat", "data/dat-files/", getClassItems(datasetName));

                // Process sampling iterations in sequential order, but each dataset runs in
                // parallel
                for (int samplingIterations = 1; samplingIterations < 100_001; samplingIterations *= 10) {
                    final int samplingIterationsFinal = samplingIterations;

                    ExecutorService executor = Executors.newFixedThreadPool(10);

                    // Process each fold in parallel using a thread pool
                    IntStream.iterate(1, foldIdx -> foldIdx + 1)
                            .limit(10)
                            .forEach(foldIdx -> {
                                final int foldIdxFinal = foldIdx;

                                executor.submit(() -> {
                                    System.out.println(
                                            "Dataset: " + datasetName + " - Iterations: " + samplingIterationsFinal);
                                    runOnFold(dataset, owa_score_function, allMeasureNames, datasetName, foldIdxFinal,
                                            samplingIterationsFinal, "results/sampling_experiment/samples/ratio_1to1/");
                                });
                            });

                    // Shutdown the executor and wait for tasks to complete
                    executor.shutdown();
                    try {
                        // Wait for all tasks to complete before proceeding
                        if (!executor.awaitTermination(12, TimeUnit.HOURS)) {
                            executor.shutdownNow();
                        }
                    } catch (InterruptedException e) {
                        executor.shutdownNow();
                        Thread.currentThread().interrupt();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace(); // Handle dataset loading exception
            }
        });
    }

    /**
     * Main entry point for the experiment.
     * 
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        runOnDataset();
    }
}
