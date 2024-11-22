package experiments.configs;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.ranking.heuristics.HighScoreSampling;

import java.io.FileReader;
import java.io.IOException;

@Getter
@Setter
public class HighScoreSamplingConfig implements QuerySelectionConfig {

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;

    // Configurable parameters
    private double noise;
    private int topK;
    private int maximumIterations;
    private NormalizationMethod normalizationMethod;

    private RankingsProvider rankingsProvider;

    /**
     * Constructor for HighScoreSamplingConfig.
     *
     * @param oracle       The oracle to use for ranking.
     * @param dataset      The dataset to use for sampling.
     * @param measureNames The measure names for normalization.
     */
    public HighScoreSamplingConfig(ArtificialOracle oracle, Dataset dataset, String[] measureNames) {
        if (oracle == null || dataset == null || measureNames == null || measureNames.length == 0) {
            throw new IllegalArgumentException("Oracle, dataset, and measure names must not be null or empty.");
        }
        setOracle(oracle);
        setDataset(dataset);
        setMeasureNames(measureNames);
    }

    @Override
    public void setUp() {
        // Initialize RankingsProvider based on configuration
        setRankingsProvider(new HighScoreSampling(this));
    }

    /**
     * Loads configurable parameters from a JSON file into the current instance.
     *
     * @param filePath The path to the JSON configuration file.
     * @throws IOException        If the file cannot be read.
     * @throws JsonSyntaxException If the JSON file has invalid syntax.
     */
    public void loadFromFile(String filePath) throws IOException, JsonSyntaxException {
        Gson gson = new Gson();

        try (FileReader reader = new FileReader(filePath)) {
            // Load the JSON configuration into a parameters object
            HighScoreSamplingParameters parameters = gson.fromJson(reader, HighScoreSamplingParameters.class);

            // Update configurable parameters
            this.noise = parameters.getNoise();
            this.topK = parameters.getTopK();
            this.maximumIterations = parameters.getMaximumIterations();
            this.normalizationMethod = parameters.getNormalizationMethodEnum();
        }
    }

    /**
     * Configurable parameters for HighScoreSampling.
     */
    @Getter
    @Setter
    private static class HighScoreSamplingParameters {
        private double noise = 0.0;
        private int topK = 2;
        private int maximumIterations = 10_000;
        private String normalizationMethod;

        /**
         * Converts the normalization method string into the corresponding enum value.
         *
         * @return NormalizationMethod value.
         */
        public NormalizationMethod getNormalizationMethodEnum() {
            try {
                return NormalizationMethod.valueOf(normalizationMethod.toUpperCase());
            } catch (IllegalArgumentException | NullPointerException e) {
                throw new IllegalArgumentException("Invalid normalization method: " + normalizationMethod, e);
            }
        }
    }
}
