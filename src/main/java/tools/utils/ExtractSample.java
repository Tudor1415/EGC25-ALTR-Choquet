package tools.utils;

import java.util.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.*;
import java.time.LocalDateTime;
import java.util.stream.Collectors;
import java.time.format.DateTimeFormatter;

import sampling.Sampler;
import tools.data.Dataset;
import sampling.BatchSampler;
import tools.rules.DecisionRule;
import tools.functions.singlevariate.*;
import tools.normalization.Normalizer.NormalizationMethod;

public class ExtractSample {

    // Possible rule measures
    public static final String CONFIDENCE = "confidence";
    public static final String LIFT = "lift";
    public static final String COSINE = "cosine";
    public static final String PHI = "phi";
    public static final String KRUSKAL = "kruskal";
    public static final String YULEQ = "yuleQ";
    public static final String ADDED_VALUE = "pavillon";
    public static final String CERTAINTY = "certainty";
    public static final String SUPPORT = "support";
    public static final String REVSUPPORT = "revsup";
    public static final String IG = "IG";

    private static String ruleToString(DecisionRule rule) {
        Set<String> antecedentValues = rule.getItemsInX();
        String consequentValues = rule.getY();

        return "[" + String.join("; ", antecedentValues) + "]" + " => " + "[" + String.join("; ", consequentValues)
                + "]";
    }

    public static void writeSampleToCSV(List<DecisionRule> rules, List<Double> scoresApprox, String fileName,
            String outputDirectory) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String directoryPath = outputDirectory + "/";
        String filePath = directoryPath + fileName + "_" + timestamp + ".csv";
        String filePathDistanceMatrix = directoryPath + fileName + "_jaccard_" + timestamp + ".csv";

        double[][] distanceMatrix = RuleUtil.computeJaccardDistance(rules);
        RuleUtil.printDistanceMatrix(distanceMatrix, filePathDistanceMatrix);
        
        try {
            // Ensure the directory exists or create it
            Files.createDirectories(Paths.get(directoryPath));

            try (FileWriter writer = new FileWriter(filePath)) {

                // If rules is empty, we only write the header
                if (!rules.isEmpty()) {
                    String[] measureNames = rules.get(0).getMeasureNames(); // Get the measure names dynamically
                    writer.append("Rule,"); // Start with "Rule,"
                    for (String measureName : measureNames) {
                        writer.append(measureName).append(","); // Append each measure name with a comma
                    }
                    writer.append("scoreApprox,\n"); // End with "scoreApprox"
                } else {
                    System.out.println("No rules available, writing empty CSV with header only.");
                    return; // Exit after writing the header
                }

                // Write data for each rule if available
                for (int i = 0; i < rules.size(); i++) {
                    writer.append(ruleToString(rules.get(i)) + ",");
                    double[] vector = rules.get(i).getAlternative().getVector();
                    for (double value : vector) {
                        writer.append(Double.toString(value)).append(",");
                    }
                    writer.append(scoresApprox.get(i) + ",");
                    writer.append("\n");
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

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
                return null;
        }
    }
    
    /**
     * Reads a dataset from the specified folder.
     * 
     * @param datasetName The name of the dataset.
     * @param folderPath  The path to the dataset.
     * @return The dataset that has been loaded.
     * @throws IOException If there is an issue reading the dataset.
     */
    public static Dataset readDataset(String datasetName, String folderPath) throws IOException {
        File folder = new File(folderPath);

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();

            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && file.getName().equals(datasetName)) {
                        // Extract the dataset name without the file extension
                        String datasetNameWithoutExtension = datasetName.replaceFirst("[.][^.]+$", "");

                        return new Dataset(file.getName(), folderPath, getClassItems(datasetNameWithoutExtension));
                    }
                }
                System.out.println("Dataset with the specified name was not found in the folder.");
            } else {
                System.out.println("The folder is empty or cannot be read.");
            }
        } else {
            System.out.println("The specified path is not a valid directory.");
        }

        return null;
    }

    private static List<Double> computeApproxScores(List<DecisionRule> sample,
            ISinglevariateFunction scoreFunction) {
        return sample.parallelStream()
                .map(rule -> scoreFunction.computeScore(rule.getAlternative()))
                .collect(Collectors.toList());
    }

    private static void processSampling(Dataset dataset, ISinglevariateFunction scoreFunction,
            String[] measureNames, String datasetName, int nbSamples, String outputDirectory, int timeoutInMinutes) {

        String datasetNameWithoutExtension = datasetName.replaceFirst("[.][^.]+$", "");

        BatchSampler batchSampler = new BatchSampler(nbSamples, dataset, scoreFunction, measureNames,
                500);

        batchSampler.setScoringFunction(scoreFunction);
        batchSampler.setNormalizationTechnique(NormalizationMethod.NO_NORMALIZATION);

        // Run the sampling with a timeout
        List<DecisionRule> sample = executeSamplingWithTimeout(batchSampler, timeoutInMinutes);

        // Process the results after sampling
        String filename = datasetNameWithoutExtension + "_" + nbSamples + "_samples";
        List<Double> approxScores = computeApproxScores(sample, scoreFunction);

        // Write the results to CSV
        writeSampleToCSV(sample, approxScores, filename, outputDirectory);
    }

    private static List<DecisionRule> executeSamplingWithTimeout(Sampler sampler, int timeoutInMinutes) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<List<DecisionRule>> future = executor.submit(() -> sampler.sample());

        List<DecisionRule> sample = new ArrayList<>();
        try {
            sample = future.get(timeoutInMinutes, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            System.err.println("Sampling timed out after " + timeoutInMinutes + " minutes.");
            future.cancel(true);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        } finally {
            executor.shutdownNow();
        }
        return sample;
    }

    /**
     * The main method to run the sampling process with command-line arguments.
     *
     * @param args Command-line arguments.
     */
    public static void main(String[] args) {
        // Check if the correct number of arguments is provided
        if (args.length < 7) {
            System.out.println("Usage: java ExtractSample <datasetName> <folderPath> <outputDirectory> "
                    + "<nbSamples> <timeoutInMinutes> <measureNames> <weights>");
            System.out.println("Possible measureNames (comma-separated): confidence, lift, cosine, phi, "
                    + "kruskal, yuleQ, pavillon, certainty, support, revsup, IG");
            System.out.println("Weights should be a comma-separated list of doubles, matching the measureNames order");
            return;
        }

        String datasetName = args[0];
        String folderPath = args[1];
        String outputDirectory = args[2];
        int nbSamples = Integer.parseInt(args[3]);
        int timeoutInMinutes = Integer.parseInt(args[4]);
        String[] measureNames = args[5].split(",");
        String[] weightsStr = args[6].split(",");

        if (measureNames.length != weightsStr.length) {
            System.err.println("The number of weights must match the number of measure names.");
            return;
        }

        double[] weights = new double[weightsStr.length];
        try {
            for (int i = 0; i < weightsStr.length; i++) {
                weights[i] = Double.parseDouble(weightsStr[i]);
            }
        } catch (NumberFormatException e) {
            System.err.println("Invalid weight format. Weights must be numbers.");
            return;
        }

        // Read the dataset
        Dataset dataset = null;
        try {
            dataset = readDataset(datasetName, folderPath);
            if (dataset == null) {
                System.err.println("Dataset could not be loaded.");
                return;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // Create the linear scoring function with the provided weights
        ISinglevariateFunction scoreFunction = new LinearScoreFunction(weights);

        // Process sampling
        processSampling(dataset, scoreFunction, measureNames, datasetName, nbSamples, outputDirectory,
                timeoutInMinutes);
    }
}