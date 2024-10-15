package experiments;

import static tools.rules.RuleMeasures.addedValue;
import static tools.rules.RuleMeasures.certainty;
import static tools.rules.RuleMeasures.confidence;
import static tools.rules.RuleMeasures.cosine;
import static tools.rules.RuleMeasures.kruskal;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import sampling.ISampler;
import sampling.SMAS;
import tools.data.Dataset;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.LinearScoreFunction;
import tools.functions.singlevariate.Choquet.ChoquetMobiusScoreFunction;
import tools.oracles.Oracle;
import tools.oracles.TicTacToeOracle;
import tools.rules.DecisionRule;
import tools.train.IterativeRankingLearn;
import tools.utils.FunctionUtil;
import tools.utils.RandomUtil;

public class InteractiveTicTacToe {
    // General settings for the experiment
    private final long seed = 548636487354L;
    private final String datasetPath = "TicTacToe/data/";

    // Learning and validation parameters
    private final int nbIterations = 15;
    private final int timeLimit = 3600;
    private final int k = 1; // K-Folds for cross-validation
    private final double trainTestRatio = 0.5; // Train-test ratio for transactions

    // Measure names and settings
    private final String[] measureNames = { confidence, kruskal, cosine, addedValue, certainty };
    private final int nbCriteria = measureNames.length;

    private final RandomUtil random = RandomUtil.getInstance();

    public static void main(String[] args) {
        try {
            new InteractiveTicTacToe().run();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Cuts the transactional dataset into K folds for cross-validation.
     *
     * @param dataset        The dataset to split.
     * @param trainTestRatio The ratio for train-test splitting.
     * @return List of datasets split into K folds.
     */
    private List<Dataset[]> cutDatasetIntoKFolds(Dataset dataset, double trainTestRatio) {
        List<Dataset[]> datasets = new ArrayList<>();
        String[][] transactionalDataset = dataset.getTransactions();
        int size = transactionalDataset.length;
        int[][] indexOfTransactionsPerFold = random.kFolds(k, size);

        for (int[] foldIndices : indexOfTransactionsPerFold) {
            List<String[]> trainTransactions = new ArrayList<>();
            List<String[]> testTransactions = new ArrayList<>();
            int trainThreshold = (int) (foldIndices.length * trainTestRatio);

            for (int i = 0; i < trainThreshold; i++) {
                trainTransactions.add(transactionalDataset[i]);
            }
            for (int i = trainThreshold; i < foldIndices.length; i++) {
                testTransactions.add(transactionalDataset[foldIndices[i]]);
            }

            Dataset trainDataset = new Dataset(trainTransactions.toArray(new String[0][0]),
                    dataset.getConsequentItemsSet());
            Dataset testDataset = new Dataset(testTransactions.toArray(new String[0][0]),
                    dataset.getConsequentItemsSet());
            datasets.add(new Dataset[] { trainDataset, testDataset });
        }
        return datasets;
    }

    /**
     * Retrieves a list of ranking learning algorithms for experimentation.
     *
     * @param oracle  The oracle representing ground truth ranking.
     * @param dataset The dataset used in the experiment.
     * @return List of ranking learning algorithms.
     */
    private List<IterativeRankingLearn> getRankingAlgorithms(Oracle oracle, Dataset dataset) {
        ISinglevariateFunction defaultFunction = new LinearScoreFunction();

        double noise = 0.0d;

        // KappalabIterative kappalab = new KappalabIterative(nbIterations,
        // new TopTwoRules(oracle, dataset, measureNames, noise), defaultFunction,
        // nbCriteria);
        // kappalab.setName(kappalab.getClass().getSimpleName());
        // kappalab.setTimeLimit(timeLimit);
        // return Collections.singletonList(kappalab);
        return null;
    }

    public void run() throws Exception {
        RandomUtil.getInstance().setSeed(seed);

        Set<String> classItemValues = new HashSet<>();
        classItemValues.add("28");
        classItemValues.add("29");

        Dataset dataset = new Dataset("tictactoe.dat", datasetPath, classItemValues);

        List<Dataset[]> fold_datasets = cutDatasetIntoKFolds(dataset, trainTestRatio);

        Oracle oracle = new TicTacToeOracle();

        int foldIdx = 1;
        for (Dataset[] fold_dataset : fold_datasets) {
            List<IterativeRankingLearn> algorithms = getRankingAlgorithms(oracle, fold_dataset[0]);

            for (IterativeRankingLearn algorithm : algorithms) {
                FunctionParameters params = algorithm.learn();

                ChoquetMobiusScoreFunction func = (ChoquetMobiusScoreFunction) FunctionUtil.getScoreFunction(params);

                System.out.println(func.getCapacity().toString());

                ISampler testSampler = new SMAS(100_000, fold_dataset[1], func, measureNames, 10);

                List<DecisionRule> bestRules = testSampler.sample();

                for (int i = 0; i < bestRules.size() - 1; i += 2) {
                    DecisionRule rule_a = bestRules.get(i);
                    DecisionRule rule_b = bestRules.get(i + 1);
                    ((TicTacToeOracle) oracle).showVisualization(rule_a, rule_b);
                }
            }

            foldIdx = foldIdx + 1;
        }
    }
}
