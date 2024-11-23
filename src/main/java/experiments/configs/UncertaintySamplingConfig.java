package experiments.configs;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import com.google.gson.Gson;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import com.google.gson.JsonSyntaxException;
import tools.ranking.heuristics.UncertaintySampling;
import tools.normalization.Normalizer.NormalizationMethod;

import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

@Getter
@Setter
public class UncertaintySamplingConfig implements QuerySelectionConfig {

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;

    // Configurable parameters
    private double noise;
    private int maximumIterations;
    private String certaintyType;
    private NormalizationMethod normalizationMethod;
    private int nbLearningIterations;
    private int changePeriod;
    private boolean startUncertainty;

    private String name = "UncertaintySampling";
    private RankingsProvider rankingsProvider;

    /**
     * Constructor for UncertaintySamplingConfig.
     *
     * @param oracle       The oracle to use for ranking.
     * @param dataset      The dataset to use for sampling.
     * @param measureNames The measure names for normalization.
     */
    public UncertaintySamplingConfig(ArtificialOracle oracle, Dataset dataset, String[] measureNames) {
        if (oracle == null || dataset == null || measureNames == null || measureNames.length == 0) {
            throw new IllegalArgumentException("Oracle, dataset, and measure names must not be null or empty.");
        }
        this.oracle = oracle;
        this.dataset = dataset;
        this.measureNames = measureNames;
    }

    @Override
    public void setUp() {
        // Initialize RankingsProvider based on configuration
        this.rankingsProvider = new UncertaintySampling(this);
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
            UncertaintySamplingParameters parameters = gson.fromJson(reader, UncertaintySamplingParameters.class);

            // Update configurable parameters
            this.noise = parameters.getNoise();
            this.maximumIterations = parameters.getMaximumIterations();
            this.certaintyType = parameters.getCertaintyType();
            this.normalizationMethod = parameters.getNormalizationMethodEnum();
            this.nbLearningIterations = parameters.getNbLearningIterations();
            this.startUncertainty = parameters.isStartUncertainty();
            this.changePeriod = parameters.getChangePeriod();
        }
    }

    /**
     * Helper class to represent configurable parameters for UncertaintySampling.
     */
    @Getter
    @Setter
    private static class UncertaintySamplingParameters {
        private double noise = 0.0;
        private int maximumIterations = 100;
        private String certaintyType = "ScoreDifference";
        private String normalizationMethod = "MIN_MAX_SCALING";
        private int nbLearningIterations = 100;
        private int changePeriod = 10;
        private boolean startUncertainty = false;

        /**
         * Converts the normalization method string into the corresponding enum value.
         *
         * @return NormalizationMethod enum value.
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
