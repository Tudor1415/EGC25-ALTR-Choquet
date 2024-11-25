package experiments.configs;

import java.util.Set;
import java.io.FileReader;
import java.nio.file.Path;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.rules.RuleMiner;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.ranking.heuristics.UncertaintyMining;
import tools.normalization.Normalizer.NormalizationMethod;

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
    private String rulesPath;
    private String dataPath;
    private String outputPath;

    private String name = "Mining";
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
        this.dataPath = dataset.getExpDir() + dataset.getFilename();

        String csvFilename = dataset.getFilename().replaceAll("\\.dat$", ".csv");
        this.outputPath = dataset.getExpDir() + rulesPath + "rules_" + csvFilename;

        // Ensure the output path exists
        Path outputDir = Paths.get(outputPath).getParent();
        if (outputDir != null && !Files.exists(outputDir)) {
            Files.createDirectories(outputDir);
        }

        Set<Integer> classItemsInt = dataset.getConsequentItemsSet().stream()
                .map(Integer::parseInt)
                .collect(Collectors.toSet());

        RuleMiner.mine(dataPath, classItemsInt, outputPath, sampleSize, randomSampleSize);

        // Initialize the rankings provider based on the current configuration
        this.rankingsProvider = new UncertaintyMining(this);
    }

    /**
     * Loads configurable parameters from a JSON file into the current instance.
     *
     * @param filePath The path to the JSON configuration file.
     * @throws IOException         If the file cannot be read.
     * @throws JsonSyntaxException If the JSON file has invalid syntax.
     */
    public void loadFromFile(String filePath) throws IOException, JsonSyntaxException {
        if (!Files.exists(Paths.get(filePath))) {
            throw new IOException("Configuration file does not exist: " + filePath);
        }

        Gson gson = new Gson();

        try (FileReader reader = new FileReader(filePath)) {
            // Load the JSON configuration into a parameters object
            MinGapsRankingParameters parameters = gson.fromJson(reader, MinGapsRankingParameters.class);

            // Update configurable parameters
            this.noise = parameters.getNoise();
            this.randomSampleSize = parameters.getRandomSampleSize();
            this.normalizationMethod = parameters.getNormalizationMethodEnum();
            this.sampleSize = parameters.getSampleSize();
            this.rulesPath = parameters.getRulesPath();
            this.minConf = parameters.getMinConf();
            this.minSup = parameters.getMinSup();
        } catch (JsonSyntaxException e) {
            throw new JsonSyntaxException("Invalid JSON syntax in file: " + filePath, e);
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
        private int sampleSize = 1000;
        private int minSup = 10;
        private int minConf = 90;
        private String rulesPath;
        private String normalizationMethod = "MIN_MAX_SCALING";

        public NormalizationMethod getNormalizationMethodEnum() {
            try {
                return NormalizationMethod.valueOf(normalizationMethod.toUpperCase());
            } catch (IllegalArgumentException | NullPointerException e) {
                throw new IllegalArgumentException("Invalid normalization method: " + normalizationMethod, e);
            }
        }
    }
}
