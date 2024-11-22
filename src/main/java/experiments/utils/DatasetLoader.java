package experiments.utils;

import java.io.File;
import java.util.Set;
import java.util.List;
import java.util.Arrays;
import java.util.HashSet;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tools.data.Dataset;

public class DatasetLoader {
    /**
     * Retrieves the class items for each dataset.
     *
     * @param datasetName The name of the dataset.
     * @return A set of class items.
     */
    public static Set<String> getClassItems(String datasetName) {
        switch (datasetName) {
            case "adult":
                return new HashSet<>(Arrays.asList("145", "146"));
            case "bank":
                return new HashSet<>(Arrays.asList("89", "90"));
            case "connect":
                return new HashSet<>(Arrays.asList("127", "128"));
            case "credit":
                return new HashSet<>(Arrays.asList("111", "112"));
            case "dota":
                return new HashSet<>(Arrays.asList("346", "347"));
            case "toms":
                return new HashSet<>(Arrays.asList("911", "912"));
            case "mushroom":
                return new HashSet<>(Arrays.asList("116", "117"));
            case "banknote":
                return new HashSet<>(Arrays.asList("17", "18"));
            case "heart":
                return new HashSet<>(Arrays.asList("32", "33"));
            case "ionosphere":
                return new HashSet<>(Arrays.asList("145", "146"));
            case "ilpd":
                return new HashSet<>(Arrays.asList("15", "16"));
            case "magic":
                return new HashSet<>(Arrays.asList("80", "81"));
            case "medical_kaggle":
                return new HashSet<>(Arrays.asList("126", "127"));
            case "parkinsons":
                return new HashSet<>(Arrays.asList("52", "53"));
            case "pima":
                return new HashSet<>(Arrays.asList("31", "32"));
            case "skin":
                return new HashSet<>(Arrays.asList("120", "121"));
            case "tictactoe":
                return new HashSet<>(Arrays.asList("28", "29"));
            case "transfusion":
                return new HashSet<>(Arrays.asList("7", "8"));
            case "travel-insurance":
                return new HashSet<>(Arrays.asList("212", "213"));
            case "twitter":
                return new HashSet<>(Arrays.asList("1512", "1513"));
            case "wdbc":
                return new HashSet<>(Arrays.asList("89", "90"));
            case "weatherAUS":
                return new HashSet<>(Arrays.asList("152", "153"));
            case "iris":
                return new HashSet<>(Arrays.asList("12", "13"));
            default:
                return new HashSet<>();
        }
    }

    private static final Logger logger = LoggerFactory.getLogger(DatasetLoader.class);

    public static List<Dataset> loadDatasetsFromDirectory(String datasetPath, String datasetName) {
        File folder = new File(datasetPath);
        List<Dataset> datasets = new ArrayList<>();

        if (!folder.exists() || !folder.isDirectory()) {
            logger.error("The specified path '{}' does not exist or is not a valid directory.", datasetPath);
            return datasets;
        }

        File[] files = folder.listFiles();
        if (files == null || files.length == 0) {
            logger.warn("The folder '{}' is empty or cannot be read.", datasetPath);
            return datasets;
        }

        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(".dat")) {
                try {
                    Dataset dataset = new Dataset(file.getName(), datasetPath,
                            getClassItems(datasetName));
                    datasets.add(dataset);
                } catch (Exception e) {
                    logger.error("Error loading dataset from file '{}': {}", file.getName(), e.getMessage());
                }
            }
        }

        logger.info("Successfully loaded {} datasets from '{}'.", datasets.size(), datasetPath);
        return datasets;
    }
}
