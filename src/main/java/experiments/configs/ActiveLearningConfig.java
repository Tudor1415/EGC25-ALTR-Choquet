package experiments.configs;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class ActiveLearningConfig {
    private String experimentConfigFolderPath;
    private List<ActiveLearningExperimentConfig> experimentConfigs;
    private int maxParallelExperiments;
}
