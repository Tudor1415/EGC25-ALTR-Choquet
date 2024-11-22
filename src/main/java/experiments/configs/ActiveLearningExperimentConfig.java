package experiments.configs;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ActiveLearningExperimentConfig {
    private String experimentName;
    private String dataDirectory;
    private String outputDirectory;
    private String loggingPath;
    private String[] datasetNames;
    private String[] measureNames;
    private String[] oracles; 
    private String[] learningToRankAlgorithms;
    private String[] querySelectionAlgorithms;
    private String[] querySelectionConfigPaths;
    private int nbLearningIterations;
    private int testSetSize;
    private int maxAntSize;
    private boolean logToFile;
    private double noiseLevel;
}
