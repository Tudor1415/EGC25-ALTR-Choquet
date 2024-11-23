package experiments.utils;

import java.util.Set;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tools.data.Dataset;
import sampling.RandomSampler;
import tools.oracles.OWAOracle;
import tools.rules.DecisionRule;
import tools.metrics.ExperimentLogger;
import tools.oracles.ArtificialOracle;
import tools.oracles.ChiSquaredOracle;
import experiments.configs.MiningConfig;
import tools.train.IterativeRankingLearn;
import tools.oracles.InformationGainOracle;
import tools.train.iterative.KappalabIterative;
import experiments.configs.QuerySelectionConfig;
import experiments.configs.HighScoreSamplingConfig;
import experiments.configs.UncertaintySamplingConfig;
import tools.functions.singlevariate.LinearScoreFunction;
import experiments.configs.ActiveLearningExperimentConfig;
import tools.normalization.Normalizer.NormalizationMethod;

/**
 * Class responsible for running an individual active learning experiment
 * based on the provided configuration.
 */
public class ActiveLearningExperimentRunner {
    private static final Logger logger = LoggerFactory.getLogger(ActiveLearningExperimentRunner.class.getSimpleName());

    private ActiveLearningExperimentConfig config;

    public ActiveLearningExperimentRunner(ActiveLearningExperimentConfig config) {
        this.config = config;
    }

    public void run() {
        ExecutorService executor = Executors.newFixedThreadPool(config.getNbParallelThreads());

        try {

            for (String datasetName : config.getDatasetNames()) {
                executor.submit(() -> {
                    try {
                        // Load all the folds
                        String dataPath = config.getDataDirectory() + datasetName;
                        List<Dataset> trainDatasets = loadDatasets(dataPath, "/train/", datasetName);
                        List<Dataset> testDatasets = loadDatasets(dataPath, "/test/", datasetName);
                        logger.info("Loaded {} train folds and {} test folds for dataset: {}",
                                trainDatasets.size(), testDatasets.size(), datasetName);

                        // Run Experiment on Each Fold in Parallel
                        int numFolds = trainDatasets.size();
                        for (int foldIdx = 0; foldIdx < numFolds; foldIdx++) {
                            final int currentFoldIdx = foldIdx;

                            Dataset trainDataset = trainDatasets.get(currentFoldIdx);
                            Dataset testDataset = testDatasets.get(currentFoldIdx);

                            // Step 2: Initialize Oracles
                            List<ArtificialOracle> trainOracles = initializeOracles(trainDataset);
                            List<ArtificialOracle> testOracles = initializeOracles(testDataset);

                            if (trainOracles.size() != testOracles.size()) {
                                throw new IllegalArgumentException(
                                        "Mismatch between the number of train and test oracles.");
                            }

                            // Run experiment for each pair of train and test oracles
                            for (int i = 0; i < trainOracles.size(); i++) {
                                ArtificialOracle trainOracle = trainOracles.get(i);
                                ArtificialOracle testOracle = testOracles.get(i);

                                // logger.info(
                                // "Starting experiments with train oracle: {} and test oracle: {} for fold
                                // {}/{} of dataset: {}",
                                // trainOracle.getTYPE(), testOracle.getTYPE(), currentFoldIdx + 1, numFolds,
                                // datasetName);

                                // Use train oracle to initialize selection strategies
                                List<QuerySelectionConfig> selectionStrategies = initializeQuerySelectionConfigs(
                                        trainOracle, trainDataset, config.getMeasureNames());

                                // Initialize learning algorithms with the selection strategies
                                List<IterativeRankingLearn> learningAlgorithms = initializeLearningAlgorithms(
                                        selectionStrategies);

                                // Use test oracle to run experiments
                                runExperimentOnFold(datasetName, trainDataset, testDataset, testOracle,
                                        learningAlgorithms,
                                        currentFoldIdx);

                                logger.info(
                                        "Completed experiments for train oracle: {} and test oracle: {} on fold {}/{} of dataset: {}",
                                        trainOracle.getTYPE(), testOracle.getTYPE(), currentFoldIdx + 1, numFolds,
                                        datasetName);
                            }

                        }
                    } catch (Exception e) {
                        logger.error("Error dataset {}: {}", datasetName, e.getMessage(), e);
                    }
                });
            }

            executor.shutdown();
            if (!executor.awaitTermination(1, TimeUnit.HOURS)) {
                executor.shutdownNow();
                logger.warn("Timeout reached. Forced shutdown of remaining tasks.");
            }

            logger.info("Experiment '{}' completed successfully.", config.getExperimentName());
        } catch (Exception e) {
            logger.error("Error running experiment '{}': {}", config.getExperimentName(), e.getMessage(), e);
        } finally {
            if (!executor.isShutdown()) {
                executor.shutdownNow();
            }
        }
    }

    private List<Dataset> loadDatasets(String dataDirectory, String subDirectory, String datasetName) throws Exception {
        String datasetPath = dataDirectory + subDirectory;
        List<Dataset> datasets = DatasetLoader.loadDatasetsFromDirectory(datasetPath, datasetName);

        if (datasets.isEmpty()) {
            throw new Exception("No datasets found in directory: " + datasetPath);
        }

        return datasets;
    }

    private List<ArtificialOracle> initializeOracles(Dataset dataset) {
        List<ArtificialOracle> oracles = new ArrayList<>();

        // Initialize oracles based on config
        for (String oracleName : config.getOracles()) {
            switch (oracleName) {
                case "ChiSquaredOracle":
                    oracles.add(new ChiSquaredOracle(dataset.getNbTransactions()));
                    break;
                case "InformationGainOracle":
                    oracles.add(new InformationGainOracle(dataset.getNbTransactions()));
                    break;
                case "OWAOracle":
                    oracles.add(new OWAOracle(0.01, config.getMeasureNames().length));
                    break;
                default:
                    logger.warn("Unknown oracle specified: {}", oracleName);
            }
        }

        return oracles;
    }

    private List<IterativeRankingLearn> initializeLearningAlgorithms(List<QuerySelectionConfig> selectionStrategies) {
        List<IterativeRankingLearn> algorithms = new ArrayList<>();

        // Ensure the number of query selection strategies matches the number of
        // learning algorithms
        if (config.getLearningToRankAlgorithms().length != selectionStrategies.size()) {
            logger.error("Mismatch between the number of learning algorithms and query selection strategies. " +
                    "Algorithms: {}, Strategies: {}",
                    config.getLearningToRankAlgorithms().length,
                    selectionStrategies.size());
            return algorithms;
        }

        // Initialize learning algorithms based on config
        for (int i = 0; i < config.getLearningToRankAlgorithms().length; i++) {
            String algorithmName = config.getLearningToRankAlgorithms()[i];
            QuerySelectionConfig queryStrategy = selectionStrategies.get(i);

            try {
                switch (algorithmName) {
                    case "KappalabIterative":
                        IterativeRankingLearn kappalab = new KappalabIterative(
                                config.getNbLearningIterations(),
                                queryStrategy.getRankingsProvider(),
                                new LinearScoreFunction(),
                                config.getMeasureNames().length);
                        kappalab.setName("KappalabIterative-" + queryStrategy.getName());
                        algorithms.add(kappalab);
                        break;

                    default:
                        logger.warn("Unknown algorithm specified: {}", algorithmName);
                }
            } catch (Exception e) {
                logger.error(
                        "Failed to initialize learning algorithm: {} with query strategy {}. Error: {}",
                        algorithmName,
                        queryStrategy.getName(),
                        e.getMessage(),
                        e);
            }
        }

        if (algorithms.isEmpty()) {
            logger.error("No learning algorithms were initialized.");
        }

        return algorithms;
    }

    private List<QuerySelectionConfig> initializeQuerySelectionConfigs(ArtificialOracle oracle, Dataset dataset,
            String[] measureNames) {
        List<QuerySelectionConfig> queryConfigs = new ArrayList<>();

        for (int i = 0; i < config.getQuerySelectionAlgorithms().length; i++) {
            String algorithmName = config.getQuerySelectionAlgorithms()[i];
            String configPath = config.getQuerySelectionConfigPaths()[i];

            try {
                QuerySelectionConfig queryConfig;

                switch (algorithmName) {
                    case "UncertaintySampling":
                        queryConfig = new UncertaintySamplingConfig(oracle, dataset, measureNames);
                        queryConfig.loadFromFile(configPath);
                        queryConfig.setUp();
                        queryConfigs.add(queryConfig);
                        break;
                    case "HighScoreSampling":
                        queryConfig = new HighScoreSamplingConfig(oracle, dataset, measureNames);
                        queryConfig.loadFromFile(configPath);
                        queryConfig.setUp();
                        queryConfigs.add(queryConfig);
                        break;
                    case "Mining":
                        queryConfig = new MiningConfig(oracle, dataset, measureNames);
                        queryConfig.loadFromFile(configPath);
                        queryConfig.setUp();
                        queryConfigs.add(queryConfig);
                        break;
                    default:
                        logger.warn("Unknown query selection algorithm: {}", algorithmName);
                        break;
                }
            } catch (Exception e) {
                logger.error("Failed to set up query selection algorithm: {} for Oracle: {} Dataset: {}. Error: {}",
                        algorithmName,
                        oracle.getClass().getSimpleName(),
                        dataset.getFilename(),
                        e.getMessage(),
                        e);
            }
        }

        return queryConfigs;
    }

    private void runExperimentOnFold(String datasetName, Dataset trainDataset, Dataset testDataset,
            ArtificialOracle oracle,
            List<IterativeRankingLearn> algorithms, int foldIdx) {
        try {
            // Step 1: Extract Test Rules
            RandomSampler sampler = new RandomSampler(testDataset, 3, 3, config.getMeasureNames(), 0.1d);
            Set<DecisionRule> sample = sampler.sample(config.getTestSetSize(),
                    testDataset.getConsequentItemsSet(), testDataset.getAntecedentItemsSet(),
                    config.getMaxAntSize());
            sampler.saveRulesToFile(sample, testDataset.getExpDir() + "test_rules_" + foldIdx + ".json");
            List<DecisionRule> testRuleList = new ArrayList<>(sample);

            logger.info("Mined {} test rules on dataset {} fold {}", testRuleList.size(), datasetName, foldIdx);
            // Step 2: Initialize Experiment Logger
            String loggingPath = config.getLoggingPath() + config.getExperimentName();
            ExperimentLogger experimentLogger = new ExperimentLogger(
                    oracle,
                    algorithms.get(0).getName(),
                    loggingPath + "/samples/",
                    datasetName,
                    foldIdx,
                    testRuleList,
                    NormalizationMethod.MIN_MAX_SCALING);

            // Step 3: Run Learning Algorithms
            for (IterativeRankingLearn algorithm : algorithms) {
                logger.info("Running algorithm: {}", algorithm.getName());

                // Attach logger to algorithm
                algorithm.addObserver(experimentLogger);

                // Run the learning process
                algorithm.learn();

                // Write iteration times
                experimentLogger.writeIterationTimes(oracle.getTYPE());

                if (algorithm instanceof KappalabIterative) {
                    String filePath = loggingPath + "/input/" + datasetName + "_" + foldIdx + "_" + algorithm.getName()
                            + ".json";
                    ((KappalabIterative) algorithm).logCurrentKappalabInput(loggingPath + "/input/", filePath);
                    logger.info("Logged KappalabIterative input for dataset: {}, fold: {}, algorithm: {}",
                            datasetName, foldIdx, algorithm.getName());
                }
            }
        } catch (Exception e) {
            logger.error("Error during fold {}: {}", foldIdx + 1, e.getMessage(), e);
        }
    }
}
