package experiments;

import java.io.File;
import java.util.Set;
import java.util.List;
import java.util.Arrays;
import java.util.HashSet;
import java.io.IOException;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import lombok.Getter;
import tools.data.Dataset;
import tools.utils.RuleUtil;
import sampling.RandomSampler;
import tools.rules.DecisionRule;
import tools.rules.DRMiningChoco;
import tools.metrics.ExperimentLogger;
import tools.oracles.ArtificialOracle;
import tools.oracles.ChiSquaredOracle;
import tools.oracles.InformationGainOracle;
import tools.ranking.heuristics.TopTwoRules;
import tools.train.iterative.KappalabIterative;
import tools.ranking.heuristics.UncertaintySampling;
import tools.functions.singlevariate.FunctionParameters;
import tools.ranking.heuristics.MinGapsRankingsProvider;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.normalization.Normalizer.NormalizationMethod;

public class ExperimentActiveLearning {
    public static final String dataDirectory = "data/folds/";
    public static final String expDirectory = "results/active_learning/samples/";

    // public static final List<String> datasetNames = Arrays.asList(
    // "bank", "credit", "dota", "toms", "connect", "mushroom", "adult",
    // "banknote", "heart", "ionosphere", "ilpd", "magic", "medical_kaggle",
    // "parkinsons", "pima", "skin", "tictactoe", "transfusion",
    // "travel-insurance", "twitter", "wdbc", "weatherAUS");

    public static final List<String> datasetNames = Arrays.asList(
            "toms", "connect", "mushroom", "adult",
            "banknote", "heart", "ionosphere", "ilpd", "magic", "medical_kaggle",
            "parkinsons", "pima", "skin", "tictactoe", "transfusion",
            "travel-insurance", "twitter", "wdbc", "weatherAUS");

    public static final @Getter String[] measureNames = { "yuleQ", "cosine", "kruskal", "pavillon", "certainty" };

    public static final int nbLearningIterations = 100;

    /**
     * Generates a list of oracles for the experiment.
     *
     * @param nbTransactions Number of transactions in the dataset.
     * @return List of oracles (comparators).
     */
    private List<ArtificialOracle> getOracles(int nbTransactions) {
        List<ArtificialOracle> oracleList = new ArrayList<>();

        ArtificialOracle InformationGain = new InformationGainOracle(nbTransactions);
        ArtificialOracle ChiSquared = new ChiSquaredOracle(nbTransactions);

        oracleList.add(ChiSquared);
        oracleList.add(InformationGain);

        return oracleList;
    }

    private void mineRulesForFold(String dataPath, Set<String> classItems, String outputCsvPath) throws Exception {
        // Convert the classItems to a set of integers
        Set<Integer> classItemsInt = classItems.stream()
                .map(Integer::parseInt)
                .collect(Collectors.toSet());

        // Mining function with minConf = 90 (for 90%) and minSup = 1
        int minConf = 90;
        int minSup = 10;

        // Perform the mining using DRMiningChoco
        DRMiningChoco.mine(dataPath, classItemsInt, outputCsvPath, minSup, minConf);

        // Print the output path after mining
        System.out.println("Mined rules have been saved to: " + outputCsvPath);

        // Optional: Verify if the file exists and print a confirmation message
        File minedRulesFile = new File(outputCsvPath);
        if (minedRulesFile.exists()) {
            System.out.println("Successfully created the mined rules file at: " + minedRulesFile.getAbsolutePath());
        } else {
            System.err.println("Failed to create the mined rules file at: " + outputCsvPath);
        }
    }

    /**
     * Retrieves a list of ranking learning algorithms for experimentation.
     *
     * @param oracle  The oracle representing ground truth ranking.
     * @param dataset The dataset used in the experiment.
     * @return List of ranking learning algorithms.
     * @throws IOException
     */
    private List<KappalabIterative> getLearningAlgorithms(ArtificialOracle oracle, Dataset dataset,
            String chocoRulesPath) throws IOException {
        double noise = 0.0d;
        List<KappalabIterative> learningAlgorithms = new ArrayList<>();

        // KappalabIterative topTwoRules = new KappalabIterative(nbLearningIterations,
        // new TopTwoRules(oracle, dataset, measureNames, noise), new
        // LinearScoreFunction(), measureNames.length);
        // topTwoRules.setName("TopTwoRules-" + noise);
        // topTwoRules.setTimeLimit(36000);
        // learningAlgorithms.add(topTwoRules);

        // KappalabIterative uncertaintySamplingSD = new
        // KappalabIterative(nbLearningIterations,
        // new UncertaintySampling(oracle, dataset, measureNames), new
        // LinearScoreFunction(), measureNames.length);
        // uncertaintySamplingSD.setName("ScoreDifference-" + noise);
        // uncertaintySamplingSD.setTimeLimit(36000);
        // learningAlgorithms.add(uncertaintySamplingSD);

        // KappalabIterative uncertaintySamplingBT = new
        // KappalabIterative(nbLearningIterations,
        // new UncertaintySampling(oracle, dataset, measureNames, "BradleyTerry"), new
        // LinearScoreFunction(),
        // measureNames.length);
        // uncertaintySamplingBT.setName("BradleyTerry-" + noise);
        // uncertaintySamplingBT.setTimeLimit(36000);
        // learningAlgorithms.add(uncertaintySamplingBT);

        KappalabIterative uncertaintySamplingTh = new KappalabIterative(nbLearningIterations,
                new UncertaintySampling(oracle, dataset, measureNames, "Thurstone"), new LinearScoreFunction(),
                measureNames.length);
        uncertaintySamplingTh.setName("Thurstone-" + noise);
        uncertaintySamplingTh.setTimeLimit(36000);
        learningAlgorithms.add(uncertaintySamplingTh);

        // DecisionRule[] minedRules = RuleUtil.extractRulesFromCSV(chocoRulesPath,
        // dataset, measureNames);

        // KappalabIterative ChoquetRank = new KappalabIterative(nbLearningIterations,
        // new MinGapsRankingsProvider(oracle, minedRules), new LinearScoreFunction(),
        // measureNames.length);
        // ChoquetRank.setName("ChoquetRank-" + noise);
        // ChoquetRank.setTimeLimit(36000);
        // learningAlgorithms.add(ChoquetRank);

        return learningAlgorithms;
    }

    /**
     * Retrieves the class items for each dataset.
     *
     * @param datasetName The name of the dataset.
     * @return A set of class items.
     */
    public static Set<String> getClassItems(String datasetName) {
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
            case "banknote":
                return new HashSet<>(Arrays.asList("17", "18"));
            case "heart":
                return new HashSet<>(Arrays.asList("32", "33"));
            case "ionosphere":
                return new HashSet<>(Arrays.asList("145", "146"));
            case "ilpd":
                return new HashSet<>(Arrays.asList("15", "16"));
            case "magic":
                return new HashSet<>(Arrays.asList("80", "81"));
            case "medical_kaggle":
                return new HashSet<>(Arrays.asList("126", "127"));
            case "parkinsons":
                return new HashSet<>(Arrays.asList("52", "53"));
            case "pima":
                return new HashSet<>(Arrays.asList("31", "32"));
            case "skin":
                return new HashSet<>(Arrays.asList("120", "121"));
            case "tictactoe":
                return new HashSet<>(Arrays.asList("28", "29"));
            case "transfusion":
                return new HashSet<>(Arrays.asList("7", "8"));
            case "travel-insurance":
                return new HashSet<>(Arrays.asList("212", "213"));
            case "twitter":
                return new HashSet<>(Arrays.asList("1512", "1513"));
            case "wdbc":
                return new HashSet<>(Arrays.asList("89", "90"));
            case "weatherAUS":
                return new HashSet<>(Arrays.asList("152", "153"));
            case "iris":
                return new HashSet<>(Arrays.asList("12", "13"));
            default:
                return null;
        }
    }

    /**
     * Reads datasets from the specified folder and only includes files with the
     * .dat extension.
     * 
     * @param datasetName The name of the dataset.
     * @param trainOrTest Specifies whether the folder contains training or testing
     *                    data.
     * @return A list of datasets that have been loaded from the .dat files.
     * @throws IOException If there is an issue reading the datasets.
     */
    public List<Dataset> readDatasetsFromFold(String datasetName, String trainOrTest) throws IOException {
        String folderPath = dataDirectory + datasetName + trainOrTest;
        File folder = new File(folderPath);

        List<Dataset> datasets = new ArrayList<>();

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();

            if (files != null) {
                for (File file : files) {
                    // Only consider files with a .dat extension
                    if (file.isFile() && file.getName().endsWith(".dat")) {
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

    private void launchExperimentOnFold(Dataset trainDataset, Dataset testDataset,
            ArtificialOracle trainOracle, ArtificialOracle testOracle, String loggingPath,
            String datasetName, int foldIdx, NormalizationMethod normMethod, List<DecisionRule> testRuleList)
            throws Exception {

        // Paths for Choco miner
        String chocoRulesPath = dataDirectory + datasetName + "/train/train_rules_" + foldIdx + ".csv";
        String foldPath = dataDirectory + datasetName + "/train/train_" + foldIdx + ".dat";

        // Mine the rules for the current fold
        // mineRulesForFold(foldPath, getClassItems(datasetName), chocoRulesPath);

        List<KappalabIterative> learningAlgorithms = getLearningAlgorithms(trainOracle,
                trainDataset, chocoRulesPath);

        learningAlgorithms.parallelStream().forEach(algorithm -> {
            try {
                System.out.println("Dataset: " + datasetName + " oracle: " + trainOracle.getTYPE()
                        + " fold: " + foldIdx + " algorithm: " + algorithm.getName());
                ExperimentLogger logger = new ExperimentLogger(testOracle, algorithm.getName(), loggingPath,
                        datasetName,
                        foldIdx,
                        testRuleList, normMethod);

                algorithm.addObserver(logger);

                FunctionParameters func = algorithm.learn();

                logger.writeIterationTimes();

                String directoryPath = "results/active_learning/input/" + datasetName + "/";
                String filename = directoryPath + algorithm.getName() + "_input_fold" + foldIdx + ".json";
                algorithm.logCurrentKappalabInput(directoryPath, filename);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

    }

    public void run() throws Exception {
        for (String datasetName : datasetNames) {
            List<Dataset> trainDatasets = readDatasetsFromFold(datasetName, "/train/");
            List<Dataset> testDatasets = readDatasetsFromFold(datasetName, "/test/");

            int foldIdx = 1;
            for (Dataset trainDataset : trainDatasets) {
                List<ArtificialOracle> trainOracles = getOracles(trainDataset.getNbTransactions());
                List<ArtificialOracle> testOracles = getOracles(testDatasets.get(foldIdx).getNbTransactions());

                // Sampling the testing set of rules
                RandomSampler sampler = new RandomSampler(testDatasets.get(foldIdx), 3, 3, getMeasureNames(), 0.1d);
                List<DecisionRule> testRuleList = new ArrayList<>(
                        sampler.sample(1_000, testDatasets.get(foldIdx).getConsequentItemsSet(),
                                testDatasets.get(foldIdx).getAntecedentItemsSet(), 10));

                String chocoRulesPath = dataDirectory + datasetName + "/train/train_rules_" + foldIdx + ".csv";

                for (int oracle_id = 0; oracle_id < trainOracles.size(); oracle_id++) {
                    List<KappalabIterative> learningAlgorithms = getLearningAlgorithms(trainOracles.get(oracle_id),
                            trainDataset, chocoRulesPath);

                    for (KappalabIterative algorithm : learningAlgorithms) {
                        System.out
                                .println("Dataset: " + datasetName + " oracle: " + trainOracles.get(oracle_id).getTYPE()
                                        + " fold: " + foldIdx + " algorithm: " + algorithm.getName());
                        ExperimentLogger logger = new ExperimentLogger(testOracles.get(oracle_id), algorithm.getName(),
                                expDirectory + datasetName + "/", datasetName, foldIdx,
                                testRuleList, NormalizationMethod.MIN_MAX_SCALING);

                        algorithm.addObserver(logger);

                        FunctionParameters func = algorithm.learn();
                    }
                }

                foldIdx++;
            }
        }
    }

    /**
     * Runs the active learning experiment in parallel, limiting to five folds
     * processed at the same time across all datasets.
     *
     * @throws Exception If an error occurs during the experiment.
     */
    public void runParallel() throws Exception {
        ExecutorService foldExecutor = Executors.newFixedThreadPool(10);

        // Iterate over each dataset
        for (String datasetName : datasetNames) {
            try {
                List<Dataset> trainDatasets = readDatasetsFromFold(datasetName, "/train/");
                List<Dataset> testDatasets = readDatasetsFromFold(datasetName, "/test/");

                // Iterate over each fold within the dataset
                for (int foldIdx = 0; foldIdx < trainDatasets.size(); foldIdx++) {
                    final int currentFoldIdx = foldIdx;
                    Dataset trainDataset = trainDatasets.get(currentFoldIdx);
                    Dataset testDataset = testDatasets.get(currentFoldIdx);

                    List<ArtificialOracle> trainOracles = getOracles(trainDataset.getNbTransactions());
                    List<ArtificialOracle> testOracles = getOracles(testDataset.getNbTransactions());

                    // Sampling the testing set of rules
                    RandomSampler sampler = new RandomSampler(testDataset, 3, 3, getMeasureNames(), 0.1d);
                    List<DecisionRule> testRuleList = new ArrayList<>(
                            sampler.sample(1_000, testDataset.getConsequentItemsSet(),
                                    testDataset.getAntecedentItemsSet(), 10));

                    // Submit a task for processing this fold
                    foldExecutor.submit(() -> {
                        try {
                            // Iterate over each oracle pair
                            for (int oracleId = 0; oracleId < trainOracles.size(); oracleId++) {
                                ArtificialOracle trainOracle = trainOracles.get(oracleId);
                                ArtificialOracle testOracle = testOracles.get(oracleId);

                                // Launch the experiment for this fold and oracle
                                launchExperimentOnFold(
                                        trainDataset,
                                        testDataset,
                                        trainOracle,
                                        testOracle,
                                        expDirectory + datasetName + "/",
                                        datasetName,
                                        currentFoldIdx + 1,
                                        NormalizationMethod.MIN_MAX_SCALING,
                                        testRuleList);
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Shutdown the executor and await termination
        foldExecutor.shutdown();
        try {
            if (!foldExecutor.awaitTermination(1, TimeUnit.HOURS)) { // Adjust timeout as needed
                foldExecutor.shutdownNow();
                if (!foldExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                    System.err.println("Executor did not terminate.");
                }
            }
        } catch (InterruptedException ie) {
            foldExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public static void main(String[] args) throws Exception {
        new ExperimentActiveLearning().run();
    }
}
