package experiments;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

import lombok.Getter;
import sampling.RandomSampler;
import tools.data.Dataset;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.metrics.ExperimentLogger;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.oracles.ChiSquaredOracle;
import tools.oracles.OWAOracle;
import tools.ranking.heuristics.MinGapsRankingsProvider;
import tools.ranking.heuristics.TopTwoRules;
import tools.ranking.heuristics.UncertaintySampling;
import tools.rules.DecisionRule;
import tools.train.IterativeRankingLearn;
import tools.train.iterative.KappalabIterative;
import tools.utils.RuleUtil;

public class ExperimentActiveLearning {
    public static final String dataDirectory = "data/folds/";
    public static final String expDirectory = "results/active_learning/samples/";

    // public static final List<String> datasetNames = Arrays.asList("adult",
    // "bank", "connect", "credit", "dota", "toms");
    public static final List<String> datasetNames = Arrays.asList("bank", "credit", "connect");

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

        ArtificialOracle OWAOracle = new OWAOracle(0.01, measureNames.length);
        ArtificialOracle ChiSquared = new ChiSquaredOracle(nbTransactions);

        oracleList.add(ChiSquared);
        oracleList.add(OWAOracle);

        return oracleList;
    }

    /**
     * Retrieves a list of ranking learning algorithms for experimentation.
     *
     * @param oracle  The oracle representing ground truth ranking.
     * @param dataset The dataset used in the experiment.
     * @return List of ranking learning algorithms.
     * @throws IOException
     */
    private List<IterativeRankingLearn> getLearningAlgorithms(ArtificialOracle oracle, Dataset dataset,
            String chocoRulesPath) throws IOException {
        double noise = 0.0d;
        List<IterativeRankingLearn> learningAlgorithms = new ArrayList<>();

        KappalabIterative topTwoRules = new KappalabIterative(nbLearningIterations,
                new TopTwoRules(oracle, dataset, measureNames, noise), new LinearScoreFunction(), measureNames.length);
        topTwoRules.setName("TopTwoRules-" + noise);
        topTwoRules.setTimeLimit(3600);
        learningAlgorithms.add(topTwoRules);

        KappalabIterative uncertaintySamplingSD = new KappalabIterative(nbLearningIterations,
                new UncertaintySampling(oracle, dataset, measureNames), new LinearScoreFunction(), measureNames.length);
        uncertaintySamplingSD.setName("ScoreDifference-" + noise);
        uncertaintySamplingSD.setTimeLimit(3600);
        learningAlgorithms.add(uncertaintySamplingSD);

        KappalabIterative uncertaintySamplingBT = new KappalabIterative(nbLearningIterations,
                new UncertaintySampling(oracle, dataset, measureNames, "BradleyTerry"), new LinearScoreFunction(),
                measureNames.length);
        uncertaintySamplingBT.setName("BradleyTerry-" + noise);
        uncertaintySamplingBT.setTimeLimit(3600);
        learningAlgorithms.add(uncertaintySamplingBT);

        KappalabIterative uncertaintySamplingTh = new KappalabIterative(nbLearningIterations,
                new UncertaintySampling(oracle, dataset, measureNames, "Thurstone"), new LinearScoreFunction(),
                measureNames.length);
        uncertaintySamplingTh.setName("Thurstone-" + noise);
        uncertaintySamplingTh.setTimeLimit(3600);
        learningAlgorithms.add(uncertaintySamplingTh);

        DecisionRule[] minedRules = RuleUtil.extractRulesFromCSV(chocoRulesPath, dataset, measureNames);

        KappalabIterative ChoquetRank = new KappalabIterative(nbLearningIterations,
                new MinGapsRankingsProvider(oracle, minedRules), new LinearScoreFunction(), measureNames.length);
        ChoquetRank.setName("ChoquetRank-" + noise);
        ChoquetRank.setTimeLimit(3600);
        learningAlgorithms.add(ChoquetRank);

        return learningAlgorithms;
    }

    /**
     * Retrieves the class items for each dataset.
     *
     * @param datasetName The name of the dataset.
     * @return A set of class items.
     */
    private Set<String> getClassItems(String datasetName) {
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

        String chocoRulesPath = dataDirectory + datasetName + "/train/train_rules_" + foldIdx + ".csv";

        List<IterativeRankingLearn> learningAlgorithms = getLearningAlgorithms(trainOracle,
                trainDataset, chocoRulesPath);

        learningAlgorithms.parallelStream().forEach(algorithm -> {
            try {
                System.out.println("Dataset: " + datasetName + " oracle: " + trainOracle.getTYPE()
                        + " fold: " + (foldIdx + 1) + " algorithm: " + algorithm.getName());
                ExperimentLogger logger = new ExperimentLogger(testOracle, algorithm.getName(), loggingPath,
                        datasetName,
                        foldIdx,
                        testRuleList, normMethod);

                algorithm.addObserver(logger);

                FunctionParameters func = algorithm.learn();
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
                    List<IterativeRankingLearn> learningAlgorithms = getLearningAlgorithms(trainOracles.get(oracle_id),
                            trainDataset, chocoRulesPath);

                    for (IterativeRankingLearn algorithm : learningAlgorithms) {
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

    public void runParallel() throws Exception {
        for (String datasetName : datasetNames) {
            try {
                List<Dataset> trainDatasets = readDatasetsFromFold(datasetName, "/train/");
                List<Dataset> testDatasets = readDatasetsFromFold(datasetName, "/test/");

                IntStream.range(0, trainDatasets.size()).parallel().forEach(foldIdx -> {
                    Dataset trainDataset = trainDatasets.get(foldIdx);
                    Dataset testDataset = testDatasets.get(foldIdx);

                    List<ArtificialOracle> trainOracles = getOracles(trainDataset.getNbTransactions());
                    List<ArtificialOracle> testOracles = getOracles(testDataset.getNbTransactions());

                    // Sampling the testing set of rules
                    RandomSampler sampler = new RandomSampler(testDatasets.get(foldIdx), 3, 3, getMeasureNames(), 0.1d);
                    List<DecisionRule> testRuleList = new ArrayList<>(
                            sampler.sample(1_000, testDatasets.get(foldIdx).getConsequentItemsSet(),
                                    testDatasets.get(foldIdx).getAntecedentItemsSet(), 10));

                    IntStream.range(0, trainOracles.size()).parallel().forEach(oracleId -> {
                        ArtificialOracle trainOracle = trainOracles.get(oracleId);
                        ArtificialOracle testOracle = testOracles.get(oracleId);
                        try {
                            launchExperimentOnFold(
                                    trainDataset,
                                    testDataset,
                                    trainOracle,
                                    testOracle,
                                    expDirectory + datasetName + "/",
                                    datasetName,
                                    foldIdx + 1,
                                    NormalizationMethod.MIN_MAX_SCALING,
                                    testRuleList);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    });
                });
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        ;
    }

    public static void main(String[] args) throws Exception {
        new ExperimentActiveLearning().runParallel();
    }
}
