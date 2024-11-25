package experiments;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
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
        String config_filepath = "experimental_configs/testing/config.json";

        try {
            ActiveLearningConfig activeLearningConfig = ConfigLoader.loadActiveLearningConfig(config_filepath);
            int maxParallelExperiments = activeLearningConfig.getMaxParallelExperiments();

            // Custom ThreadFactory to set thread priority
            ThreadFactory threadFactory = new ThreadFactory() {
                private final ThreadFactory defaultFactory = Executors.defaultThreadFactory();

                @Override
                public Thread newThread(Runnable r) {
                    Thread thread = defaultFactory.newThread(r);
                    thread.setPriority(Thread.MAX_PRIORITY-1);
                    return thread;
                }
            };

            // Create an ExecutorService with the custom ThreadFactory
            ExecutorService executorService = Executors.newFixedThreadPool(maxParallelExperiments, threadFactory);

            List<ActiveLearningExperimentConfig> experimentConfigs = activeLearningConfig.getExperimentConfigs();

            for (ActiveLearningExperimentConfig experimentConfig : experimentConfigs) {
                executorService.submit(() -> {
                    runExperiment(experimentConfig);
                });
            }

            // Initiate shutdown after submitting all tasks
            executorService.shutdown();

            // // Wait for up to 1 hour for tasks to complete
            // if (!executorService.awaitTermination(2, TimeUnit.HOURS)) {
            //     executorService.shutdownNow(); // Force shutdown if tasks are still running
            //     logger.warn("Timeout reached. Forced shutdown of remaining tasks.");
            // } else {
            //     logger.info("All experiments completed successfully within the timeout.");
            // }

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
