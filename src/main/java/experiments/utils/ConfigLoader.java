package experiments.utils;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import experiments.configs.ActiveLearningConfig;
import experiments.configs.ActiveLearningExperimentConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for loading Active Learning Configurations from JSON files.
 */
public class ConfigLoader {

    private static final Logger logger = LoggerFactory.getLogger(ConfigLoader.class);

    /**
     * Loads an ActiveLearningConfig object from a folder path containing experiment configuration JSON files.
     *
     * @param folderPath the path to the folder containing JSON configuration files
     * @return an ActiveLearningConfig object
     * @throws IOException        if there is an error reading the files
     * @throws JsonSyntaxException if a JSON file is not valid
     */
    public static ActiveLearningConfig loadActiveLearningConfig(String folderPath) throws IOException, JsonSyntaxException {
        Gson gson = new Gson();
        ActiveLearningConfig config = new ActiveLearningConfig();
        config.setExperimentConfigFolderPath(folderPath);

        List<ActiveLearningExperimentConfig> experimentConfigs = new ArrayList<>();

        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            logger.error("The provided folder path does not exist or is not a directory: {}", folderPath);
            throw new IllegalArgumentException("Invalid folder path: " + folderPath);
        }

        File[] files = folder.listFiles((dir, name) -> name.endsWith(".json"));
        if (files == null || files.length == 0) {
            logger.error("No JSON files found in the specified folder: {}", folderPath);
            throw new IOException("No JSON files in folder: " + folderPath);
        }

        logger.info("Found {} JSON files in folder: {}", files.length, folderPath);

        for (File file : files) {
            try (FileReader reader = new FileReader(file)) {
                ActiveLearningExperimentConfig experimentConfig = gson.fromJson(reader, ActiveLearningExperimentConfig.class);
                experimentConfigs.add(experimentConfig);
                logger.info("Successfully loaded config from file: {}", file.getName());
            } catch (JsonSyntaxException e) {
                logger.error("Invalid JSON syntax in file: {}. Error: {}", file.getName(), e.getMessage());
                throw e;
            } catch (IOException e) {
                logger.error("Error reading file: {}. Error: {}", file.getName(), e.getMessage());
                throw e;
            }
        }

        config.setExperimentConfigs(experimentConfigs);
        return config;
    }
}
