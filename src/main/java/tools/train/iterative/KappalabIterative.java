package tools.train.iterative;

import java.io.File;
import java.util.List;
import java.io.FileWriter;
import java.util.ArrayList;
import java.io.IOException;
import java.io.BufferedWriter;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.CancellationException;

import com.google.gson.Gson;

import lombok.Setter;
import tools.ranking.Ranking;
import tools.utils.FunctionUtil;
import tools.ranking.RankingsProvider;
import tools.alternatives.IAlternative;
import tools.train.IterativeRankingLearn;
import tools.utils.kappalab.KappalabInput;
import tools.utils.kappalab.KappalabUtils;
import tools.utils.kappalab.KappalabOutput;
import tools.utils.kappalab.KappalabRScriptCaller;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.Choquet.ChoquetMobiusScoreFunction;

/**
 * Kappalab Iterative is a learning class that communicates with an R script
 * using JSON files.
 *
 * @param nbIterations     Number of iterations for the iterative learning
 *                         process.
 * @param rankingsProvider The provider of rankings used for learning (e.g., the
 *                         heuristic).
 * @param func             The score function representing the current state of
 *                         learning (e.g., the Choquet integral).
 * @param nbMeasures       Number of measures/criteria in the ranking.
 */
public class KappalabIterative extends IterativeRankingLearn {

    @Setter
    private double delta = 1e-6d;
    @Setter
    private int kAdditivity = 2;
    @Setter
    private String approachType = "Generalized Least Squares";

    // Class variables to store the last KappalabInput and FunctionParameters
    private KappalabInput lastKappalabInput;
    private FunctionParameters lastFunctionParameters;

    public KappalabIterative(int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func,
            int nbMeasures) {
        super(nbIterations, rankingsProvider, func, nbMeasures);
    }

    /**
     * Logs the current KappalabInput to a specified directory as a JSON file.
     * Creates the directory if it does not exist.
     */
    public void logCurrentKappalabInput(String directoryPath, String filePath) throws IOException {
        if (lastKappalabInput == null) {
            System.err.println("No KappalabInput available to log.");
            return;
        }

        logObjectToFile(directoryPath, filePath, lastKappalabInput);
    }

    /**
     * Logs the current FunctionParameters to a specified directory as a JSON file.
     * Creates the directory if it does not exist.
     */
    public void logCurrentFunctionParameters(String directoryPath, String filePath) throws IOException {
        if (lastFunctionParameters == null) {
            System.err.println("No FunctionParameters available to log.");
            return;
        }

        logObjectToFile(directoryPath, filePath, lastFunctionParameters);
    }

    /**
     * Helper method to log an object to a JSON file.
     */
    private void logObjectToFile(String directoryPath, String filePath, Object object) throws IOException {
        // Ensure the directory exists
        File directory = new File(directoryPath);
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Failed to create directory: " + directoryPath);
        }

        // Create the JSON file
        File jsonFile = new File(filePath);

        // Write the object to the JSON file
        Gson gson = new Gson();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(jsonFile))) {
            writer.write(gson.toJson(object));
        }

        System.out.println("Logged to: " + filePath);
    }

    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        List<Ranking<IAlternative>> rankingsCopy = new ArrayList<>(rankings);

        while (!rankingsCopy.isEmpty()) {
            KappalabInput input = new KappalabInput(kAdditivity, approachType);

            for (Ranking<IAlternative> ranking : rankingsCopy) {
                KappalabUtils.addRankingToKappalabInput(ranking, input, delta);
            }

            lastKappalabInput = input;

            File inputFile = File.createTempFile("kappalab_input", ".json");
            File outputFile = File.createTempFile("kappalab_output", ".json");

            KappalabRScriptCaller kappalabRScript = new KappalabRScriptCaller(inputFile, outputFile, input);

            ExecutorService executor = Executors.newSingleThreadExecutor();
            long start = System.currentTimeMillis();

            Future<KappalabOutput> res = executor.submit(kappalabRScript);

            try {
                KappalabOutput output = timeLimit == 0 ? res.get() : res.get(timeRemaining, TimeUnit.MILLISECONDS);
                long time = System.currentTimeMillis() - start;
                timeRemaining -= time;

                if (output.getErrorMessages() != null) {
                    return FunctionUtil.logErrorFunction(output.getErrorMessages());
                }

                lastFunctionParameters = FunctionUtil.getFunctionParameters(
                        ChoquetMobiusScoreFunction.TYPE, nbMeasures, kAdditivity, output.getCapacities(), time / 1000d);

                return lastFunctionParameters;

            } catch (TimeoutException e) {
                String[] errorMessages = { "Timeout while waiting for Kappalab R script to finish." };
                return FunctionUtil.logErrorFunction(errorMessages);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                String[] errorMessages = { "Execution was interrupted." };
                return FunctionUtil.logErrorFunction(errorMessages);

            } catch (ExecutionException | CancellationException e) {
                String causeMessage = e.getCause() != null ? e.getCause().getMessage() : "Unknown error";
                String[] errorMessages = { "Execution or cancellation error: " + causeMessage };
                return FunctionUtil.logErrorFunction(errorMessages);

            } finally {
                executor.shutdown();
            }
        }

        String[] errorMessages = { "Failed to learn from rankings. All alternatives have been removed." };
        return FunctionUtil.logErrorFunction(errorMessages);
    }
}
