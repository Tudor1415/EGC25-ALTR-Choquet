package experiments.utils;

import java.util.ArrayList;
import java.util.List;

import javax.xml.crypto.Data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import experiments.configs.ActiveLearningExperimentConfig;
import experiments.configs.HighScoreSamplingConfig;
import experiments.configs.MiningConfig;
import experiments.configs.QuerySelectionConfig;
import experiments.configs.UncertaintySamplingConfig;
import tools.data.Dataset;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.metrics.ExperimentLogger;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.oracles.ChiSquaredOracle;
import tools.oracles.InformationGainOracle;
import tools.oracles.OWAOracle;
import tools.ranking.heuristics.UncertaintySampling;
import tools.rules.DecisionRule;
import tools.rules.RuleMiner;
import tools.train.IterativeRankingLearn;
import tools.train.iterative.KappalabIterative;

/**
 * Class responsible for running an individual active learning experiment
 * based on the provided configuration.
 */
public class ActiveLearningExperimentRunner {
    private static final Logger logger = LoggerFactory.getLogger(ActiveLearningExperimentRunner.class);

    private ActiveLearningExperimentConfig config;

    public ActiveLearningExperimentRunner(ActiveLearningExperimentConfig config) {
        this.config = config;
    }

    public void run() {
        try {
            logger.info("Starting Experiment: {}", config.getExperimentName());

            // Step 1: Load all the folds
            List<Dataset> trainDatasets = loadDatasets(config.getDataDirectory(), "/train/");
            List<Dataset> testDatasets = loadDatasets(config.getDataDirectory(), "/test/");

            // Step 4: Run Experiment on Each Fold
            int numFolds = trainDatasets.size();
            for (int foldIdx = 0; foldIdx < numFolds; foldIdx++) {
                Dataset trainDataset = trainDatasets.get(foldIdx);
                Dataset testDataset = testDatasets.get(foldIdx);

                // Step 2: Initialize Oracles
                List<ArtificialOracle> oracles = initializeOracles(testDataset);

                logger.info("Processing fold {}/{}", foldIdx + 1, numFolds);

                // Run experiment for each oracle
                for (ArtificialOracle oracle : oracles) {
                    List<QuerySelectionConfig> selectionStrategies = initializeQuerySelectionConfigs(oracle,
                            trainDataset,
                            config.getMeasureNames());
                    List<IterativeRankingLearn> learningAlgorithms = initializeLearningAlgorithms(selectionStrategies);
                    runExperimentOnFold(trainDataset, testDataset, oracle, learningAlgorithms, foldIdx);
                }
            }

            logger.info("Experiment '{}' completed successfully.", config.getExperimentName());
        } catch (Exception e) {
            logger.error("Error running experiment '{}': {}", config.getExperimentName(), e.getMessage(), e);
        }
    }

    private List<Dataset> loadDatasets(String dataDirectory, String subDirectory) throws Exception {
        String datasetPath = dataDirectory + subDirectory;
        logger.info("Loading datasets from: {}", datasetPath);

        // Use DatasetLoader to load datasets (implement this utility as needed)
        List<Dataset> datasets = DatasetLoader.loadDatasetsFromDirectory(
                datasetPath, config.getDatasetNames());

        if (datasets.isEmpty()) {
            throw new Exception("No datasets found in directory: " + datasetPath);
        }

        logger.info("Loaded {} datasets from {}", datasets.size(), datasetPath);
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

        logger.info("Initialized {} oracles.", oracles.size());
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
                        logger.info("Initialized KappalabIterative algorithm with query strategy {}", queryStrategy.getName());
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
        } else {
            logger.info("Successfully initialized {} learning algorithms.", algorithms.size());
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

    private void runExperimentOnFold(Dataset trainDataset, Dataset testDataset, ArtificialOracle oracle,
            List<IterativeRankingLearn> algorithms, int foldIdx) {
        try {
            logger.info("Running experiment on fold {} with oracle {}", foldIdx + 1, oracle.getTYPE());

            // Step 1: Mine Rules
            List<DecisionRule> minedRules = RuleMiner.mineRulesForDataset(
                    trainDataset, config.getMeasureNames(), config.getMinSupport(), config.getMinConfidence());

            // Step 2: Initialize Experiment Logger
            ExperimentLogger experimentLogger = new ExperimentLogger(
                    oracle,
                    algorithms.get(0).getName(),
                    config.getLoggingPath(),
                    config.getExperimentName(),
                    foldIdx,
                    minedRules,
                    NormalizationMethod.MIN_MAX_SCALING);

            // Step 3: Run Learning Algorithms
            for (IterativeRankingLearn algorithm : algorithms) {
                logger.info("Running algorithm: {}", algorithm.getName());

                // Set up query selection strategy
                UncertaintySampling querySelection = new UncertaintySampling(
                        oracle,
                        trainDataset,
                        config.getMeasureNames());
                algorithm.setQuerySelectionStrategy(querySelection);

                // Attach logger to algorithm
                algorithm.addObserver(experimentLogger);

                // Run the learning process
                algorithm.learn();

                // Write iteration times
                experimentLogger.writeIterationTimes(oracle.getTYPE());

                // Optionally, save results or model parameters
                // algorithm.saveModel(...);
            }
        } catch (Exception e) {
            logger.error("Error during fold {}: {}", foldIdx + 1, e.getMessage(), e);
        }
    }
}
