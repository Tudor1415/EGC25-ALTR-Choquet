package experiments;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import experiments.utils.ConfigLoader;
import experiments.configs.ActiveLearningConfig;
import experiments.utils.ActiveLearningExperimentRunner;
import experiments.configs.ActiveLearningExperimentConfig;

public class ActiveLearningExperiment {
    private static final Logger logger = LoggerFactory.getLogger(ActiveLearningExperiment.class.getSimpleName());

    public static void main(String[] args) {
        String config_filepath = "experimental_configs/active_learning/config.json";

        try {
            ActiveLearningConfig activeLearningConfig = ConfigLoader.loadActiveLearningConfig(config_filepath);
            int maxParallelExperiments = activeLearningConfig.getMaxParallelExperiments();

            ExecutorService executorService = Executors.newFixedThreadPool(maxParallelExperiments);
            List<ActiveLearningExperimentConfig> experimentConfigs = activeLearningConfig.getExperimentConfigs();
            long waitTimeBetweenExperiments = activeLearningConfig.getWaitTimeBetweenExp(); // Time in milliseconds

            for (ActiveLearningExperimentConfig experimentConfig : experimentConfigs) {
                executorService.submit(() -> {
                    runExperiment(experimentConfig);
                });

                // Wait for the specified time before launching the next experiment
                try {
                    logger.info("Waiting for {} milliseconds before launching the next experiment.", waitTimeBetweenExperiments);
                    Thread.sleep(waitTimeBetweenExperiments);
                } catch (InterruptedException e) {
                    logger.warn("Thread interrupted while waiting between experiments: {}", e.getMessage());
                    Thread.currentThread().interrupt(); // Restore the interrupted status
                }
            }

            executorService.shutdown();

        } catch (Exception e) {
            logger.error("Failed to load configurations: {}", e.getMessage(), e);
        }
    }

    private static void runExperiment(ActiveLearningExperimentConfig experimentConfig) {
        try {
            logger.info("Starting Experiment: {}", experimentConfig.getExperimentName());

            ActiveLearningExperimentRunner experimentRunner = new ActiveLearningExperimentRunner(experimentConfig);
            experimentRunner.run();

            logger.info("Completed Experiment: {}", experimentConfig.getExperimentName());
        } catch (OutOfMemoryError e) {
            logger.error("OutOfMemoryError in experiment {}: {}", experimentConfig.getExperimentName(), e.getMessage());
        } catch (Exception e) {
            logger.error("Error running experiment {}: {}", experimentConfig.getExperimentName(), e.getMessage(), e);
        }
    }

    private static void logMemoryUsage(String context) {
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        long freeMemory = runtime.freeMemory();

        logger.info("{} - Used Memory: {} MB, Free Memory: {} MB, Max Memory: {} MB",
                context,
                usedMemory / (1024 * 1024),
                freeMemory / (1024 * 1024),
                maxMemory / (1024 * 1024));
    }
}
