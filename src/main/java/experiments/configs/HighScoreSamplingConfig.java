package experiments.configs;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.ranking.RankingsProvider;
import tools.ranking.heuristics.HighScoreSampling;

@Getter
@Setter
public class HighScoreSamplingConfig implements QuerySelectionConfig {

    // Configurable parameters
    private double noise = 0.0;
    private int topK = 2;
    private int maximumIterations = 10_000;
    private NormalizationMethod normalizationMethod = NormalizationMethod.MIN_MAX_SCALING;

    // Required components
    private ArtificialOracle oracle;
    private Dataset dataset;
    private String[] measureNames;

    private RankingsProvider rankingsProvider;

    public HighScoreSamplingConfig(ArtificialOracle oracle, Dataset dataset, String[] measureNames) {
        setOracle(oracle);
        setDataset(dataset);
        setMeasureNames(measureNames);
        
        setRankingsProvider(new HighScoreSampling(this));
    }

    @Override
    public void setUp() {
    }
}
