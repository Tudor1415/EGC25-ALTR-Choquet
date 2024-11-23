package experiments.utils;

import java.io.File;
import java.util.List;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tools.data.Dataset;
import tools.utils.RuleUtil;
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
                // Load all the folds for the current dataset
                String dataPath = config.getDataDirectory() + datasetName;
                List<Dataset> trainDatasets = loadDatasets(dataPath, "/train/", datasetName);
                List<Dataset> testDatasets = loadDatasets(dataPath, "/test/", datasetName);
                // logger.info("Loaded {} train folds and {} test folds for dataset: {}",
                // trainDatasets.size(), testDatasets.size(), datasetName);

                int numFolds = trainDatasets.size();

                // Submit each fold as an independent task
                for (int foldIdx = 0; foldIdx < numFolds; foldIdx++) {
                    final int currentFoldIdx = foldIdx;

                    executor.submit(() -> {
                        try {
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

                                // Initialize selection strategies using the train oracle
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

                        } catch (Exception e) {
                            logger.error("Error processing fold {}/{} for dataset {}: {}",
                                    currentFoldIdx + 1, numFolds, datasetName, e.getMessage(), e);
                        }
                    });
                }
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
        synchronized (DatasetLoader.class) {
            String datasetPath = dataDirectory + subDirectory;
            List<Dataset> datasets = DatasetLoader.loadDatasetsFromDirectory(datasetPath, datasetName);

            if (datasets.isEmpty()) {
                throw new Exception("No datasets found in directory: " + datasetPath);
            }
            return datasets;
        }
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
                    case "UncertaintyMining":
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

    /**
     * Retrieves a list of test rules, ensuring no duplicates with previously mined
     * rules.
     * Thread-safe implementation.
     *
     * @param testDataset The dataset to mine rules from.
     * @param foldIdx     The index of the current fold.
     * @return A list of unique DecisionRule objects.
     */
    public List<DecisionRule> getTestRules(Dataset testDataset, int foldIdx) {
        String ruleOutputPath = testDataset.getExpDir() + "test_rules_" + foldIdx + ".csv";
        File ruleFile = new File(ruleOutputPath);

        // Use a synchronized block for file operations to ensure thread safety
        synchronized (ActiveLearningExperimentRunner.class) {
            List<DecisionRule> ruleList = new ArrayList<>();

            // Load rules from file if it exists
            if (ruleFile.exists()) {
                try {
                    ruleList = RuleUtil.extractRulesListFromCSV(ruleOutputPath, testDataset, config.getMeasureNames());
                    if (ruleList.size() < config.getTestSetSize()) {
                        logger.warn("Insufficient rules in file {}. Generating additional rules.", ruleOutputPath);
                    }
                } catch (IOException | ArrayIndexOutOfBoundsException e) {
                    logger.error("Error loading rules from file {}. Generating new rules.", ruleOutputPath);
                    ruleList.clear(); // Clear any partially loaded rules
                }
            }

            // Generate rules if file does not exist or rules are insufficient
            if (ruleList.size() < config.getTestSetSize()) {
                RandomSampler sampler = new RandomSampler(testDataset, 3, 3, config.getMeasureNames(), 0.1d);
                ruleList = new ArrayList<>(sampler.sample(
                        config.getTestSetSize(),
                        testDataset.getConsequentItemsSet(),
                        testDataset.getAntecedentItemsSet(),
                        config.getMaxAntSize()));
            }

            RuleUtil.saveRulesToCSV(ruleList, ruleOutputPath);

            return ruleList;
        }
    }

    private void runExperimentOnFold(String datasetName, Dataset trainDataset, Dataset testDataset,
            ArtificialOracle oracle, List<IterativeRankingLearn> algorithms, int foldIdx) {
        try {
            // Step 1: Extract Test Rules
            List<DecisionRule> testRuleList = getTestRules(testDataset, foldIdx);
            logger.info("Loaded {} test rules on dataset {} fold {}",
                    testRuleList.size(), datasetName, foldIdx);

            // Step 2: Run Learning Algorithms in Parallel
            algorithms.parallelStream().forEach(algorithm -> {
                try {
                    // Step 2.1: Initialize Experiment Logger for the current algorithm
                    String loggingPath = config.getLoggingPath() + config.getExperimentName();
                    ExperimentLogger experimentLogger = new ExperimentLogger(
                            oracle,
                            algorithm.getName(),
                            loggingPath + "/samples/",
                            datasetName,
                            foldIdx,
                            testRuleList,
                            NormalizationMethod.MIN_MAX_SCALING);

                    logger.info("{} || Running algorithm: {} on dataset {} fold {}", config.getExperimentName(),
                            algorithm.getName(), datasetName, foldIdx);

                    // Attach logger to the current algorithm
                    algorithm.addObserver(experimentLogger);

                    // Step 2.2: Run the learning process
                    algorithm.learn();

                    // Write iteration times for the current algorithm
                    experimentLogger.writeIterationTimes(oracle.getTYPE());

                    // Step 2.3: Handle specific behavior for KappalabIterative
                    if (algorithm instanceof KappalabIterative) {
                        String filePath = loggingPath + "/input/" + datasetName + "_" + foldIdx + "_"
                                + algorithm.getName()
                                + ".json";
                        ((KappalabIterative) algorithm).logCurrentKappalabInput(loggingPath + "/input/", filePath);
                        logger.info("Logged KappalabIterative input for dataset: {}, fold: {}, algorithm: {}",
                                datasetName, foldIdx, algorithm.getName());
                    }
                } catch (Exception e) {
                    logger.error("Error occurred while running algorithm: {} on dataset {} fold {}",
                            algorithm.getName(), datasetName, foldIdx, e);
                }
            });

        } catch (Exception e) {
            logger.error("Error during fold {}: {}", foldIdx + 1, e.getMessage(), e);
        }
    }
}
