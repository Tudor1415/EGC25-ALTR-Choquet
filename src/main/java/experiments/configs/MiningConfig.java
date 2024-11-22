package experiments.configs;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.rules.DecisionRule;
import tools.rules.RuleMiner;
import tools.ranking.RankingsProvider;
import tools.ranking.heuristics.UncertaintyMining;

import java.io.FileReader;
import java.io.IOException;
import java.util.Set;
import java.util.stream.Collectors;

@Getter
@Setter
public class MiningConfig implements QuerySelectionConfig {

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;

    // Configurable parameters
    private double noise = 0.0;
    private int sampleSize = 10_000;
    private int randomSampleSize = 50;
    private int minSup = 10;
    private int minConf = 90;
    private NormalizationMethod normalizationMethod = NormalizationMethod.MIN_MAX_SCALING;
    private String minedRulesOutputPath;

    // Optional: Directly set the sample
    private DecisionRule[] sample;

    private RankingsProvider rankingsProvider;

    /**
     * Constructor for MiningConfig.
     *
     * @param oracle       The oracle to use for ranking.
     * @param dataset      The dataset to use for sampling.
     * @param measureNames The measure names for normalization.
     */
    public MiningConfig(ArtificialOracle oracle, Dataset dataset, String[] measureNames) {
        if (oracle == null || dataset == null || measureNames == null || measureNames.length == 0) {
            throw new IllegalArgumentException("Oracle, dataset, and measure names must not be null or empty.");
        }
        this.oracle = oracle;
        this.dataset = dataset;
        this.measureNames = measureNames;
    }

    @Override
    public void setUp() throws Exception {
        // Initialize the rankings provider based on the current configuration
        this.rankingsProvider = new UncertaintyMining(this);
        
        String dataPath = dataset.getExpDir() + dataset.getFilename();
        
        Set<Integer> classItemsInt = dataset.getConsequentItemsSet().stream()
                .map(Integer::parseInt)
                .collect(Collectors.toSet());

        RuleMiner.mine(dataPath, classItemsInt, minedRulesOutputPath, sampleSize, randomSampleSize);
    }

    /**
     * Loads configurable parameters from a JSON file into the current instance.
     *
     * @param filePath The path to the JSON configuration file.
     * @throws IOException         If the file cannot be read.
     * @throws JsonSyntaxException If the JSON file has invalid syntax.
     */
    public void loadFromFile(String filePath) throws IOException, JsonSyntaxException {
        Gson gson = new Gson();

        try (FileReader reader = new FileReader(filePath)) {
            // Load the JSON configuration into a parameters object
            MinGapsRankingParameters parameters = gson.fromJson(reader, MinGapsRankingParameters.class);

            // Update configurable parameters
            this.noise = parameters.getNoise();
            this.randomSampleSize = parameters.getRandomSampleSize();
            this.normalizationMethod = parameters.getNormalizationMethodEnum();
            this.sampleSize = parameters.getSampleSize();
            this.minedRulesOutputPath = parameters.getRulesPath();
        }
    }

    /**
     * Helper class to represent configurable parameters for MinGapsRanking.
     */
    @Getter
    @Setter
    private static class MinGapsRankingParameters {
        private double noise = 0.0;
        private int randomSampleSize = 50;
        private String normalizationMethod = "MIN_MAX_SCALING";
        private int sampleSize = 1000;
        private String rulesPath;

        public NormalizationMethod getNormalizationMethodEnum() {
            try {
                return NormalizationMethod.valueOf(normalizationMethod.toUpperCase());
            } catch (IllegalArgumentException | NullPointerException e) {
                throw new IllegalArgumentException("Invalid normalization method: " + normalizationMethod, e);
            }
        }
    }
}
