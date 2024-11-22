package experiments.utils;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import experiments.configs.ActiveLearningConfig;
import experiments.configs.ActiveLearningExperimentConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Utility class for loading Active Learning Configurations from JSON files.
 */
public class ConfigLoader {

    private static final Logger logger = LoggerFactory.getLogger(ConfigLoader.class.getSimpleName());

    /**
     * Loads an ActiveLearningConfig object from a given filepath and its associated experiment configuration folder.
     *
     * @param filepath the path to the JSON file containing the main ActiveLearningConfig
     * @return an ActiveLearningConfig object
     * @throws IOException        if there is an error reading the files
     * @throws JsonSyntaxException if a JSON file is not valid
     */
    public static ActiveLearningConfig loadActiveLearningConfig(String filepath) throws IOException, JsonSyntaxException {
        Gson gson = new Gson();
        ActiveLearningConfig activeLearningConfig;

        // Step 1: Load the main ActiveLearningConfig.json file
        File configFile = new File(filepath);
        if (!configFile.exists() || !configFile.isFile()) {
            logger.error("ActiveLearningConfig file not found at path: {}", filepath);
            throw new IOException("ActiveLearningConfig file is missing at the specified filepath.");
        }

        try (FileReader reader = new FileReader(configFile)) {
            activeLearningConfig = gson.fromJson(reader, ActiveLearningConfig.class);
            logger.info("Successfully loaded main ActiveLearningConfig from file: {}", filepath);
        } catch (JsonSyntaxException e) {
            logger.error("Invalid JSON syntax in ActiveLearningConfig file. Error: {}", e.getMessage());
            throw e;
        } catch (IOException e) {
            logger.error("Error reading ActiveLearningConfig file. Error: {}", e.getMessage());
            throw e;
        }

        // Step 2: Load all experiment configuration files from the folder specified in the config
        String folderPath = activeLearningConfig.getExperimentConfigFolderPath();
        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            logger.error("The provided experiment config folder path does not exist or is not a directory: {}", folderPath);
            throw new IllegalArgumentException("Invalid experiment config folder path: " + folderPath);
        }

        List<ActiveLearningExperimentConfig> experimentConfigs = new ArrayList<>();
        File[] files = folder.listFiles((dir, name) -> name.endsWith(".json"));
        if (files == null || files.length == 0) {
            logger.warn("No experiment JSON files found in the specified folder: {}", folderPath);
        } else {
            logger.info("Found {} experiment JSON files in folder: {}", files.length, folderPath);
            for (File file : files) {
                try (FileReader reader = new FileReader(file)) {
                    ActiveLearningExperimentConfig experimentConfig = gson.fromJson(reader, ActiveLearningExperimentConfig.class);
                    experimentConfigs.add(experimentConfig);
                    logger.info("Successfully loaded experiment config from file: {}", file.getName());
                } catch (JsonSyntaxException e) {
                    logger.error("Invalid JSON syntax in file: {}. Error: {}", file.getName(), e.getMessage());
                    throw e;
                } catch (IOException e) {
                    logger.error("Error reading file: {}. Error: {}", file.getName(), e.getMessage());
                    throw e;
                }
            }
        }

        // Step 3: Set the experiment configs to the main config object
        activeLearningConfig.setExperimentConfigs(experimentConfigs);

        return activeLearningConfig;
    }
}
