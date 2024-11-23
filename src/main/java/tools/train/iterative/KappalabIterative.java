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

    // Class variable to store the last KappalabInput
    private KappalabInput lastKappalabInput;

    public KappalabIterative(int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func,
                             int nbMeasures) {
        super(nbIterations, rankingsProvider, func, nbMeasures);
    }

    /**
     * Logs the current KappalabInput to a specified directory as a JSON file.
     * Creates the directory if it does not exist.
     *
     * @param directoryPath The directory where the input JSON file should be saved.
     * @param filePath      The full file path where the JSON should be saved.
     * @throws IOException If an error occurs during file writing.
     */
    public void logCurrentKappalabInput(String directoryPath, String filePath) throws IOException {
        if (lastKappalabInput == null) {
            System.err.println("No KappalabInput available to log.");
            return;
        }

        // Ensure the directory exists
        File directory = new File(directoryPath);
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("Failed to create directory: " + directoryPath);
            }
        }

        // Create the JSON file
        File jsonFile = new File(filePath);

        // Write the KappalabInput to the JSON file
        Gson gson = new Gson();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(jsonFile))) {
            writer.write(gson.toJson(lastKappalabInput));
        }

        System.out.println("Kappalab input logged to: " + filePath);
    }

    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        // Create a copy of the rankings to modify if needed
        List<Ranking<IAlternative>> rankingsCopy = new ArrayList<>(rankings);
    
        while (!rankingsCopy.isEmpty()) {
            KappalabInput input = new KappalabInput(kAdditivity, approachType);
    
            // Add each ranking to the Kappalab input
            for (Ranking<IAlternative> ranking : rankingsCopy) {
                KappalabUtils.addRankingToKappalabInput(ranking, input, delta);
            }
    
            // Store the input as the last KappalabInput
            lastKappalabInput = input;
    
            // Create temporary files to store Kappalab input and output data
            File inputFile = File.createTempFile("kappalab_input", ".json");
            File outputFile = File.createTempFile("kappalab_output", ".json");
    
            // We use this to call the R script
            KappalabRScriptCaller kappalabRScript = new KappalabRScriptCaller(inputFile, outputFile, input);
    
            // Create a single-threaded executor for running the Kappalab R script
            ExecutorService executor = Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable);
                thread.setPriority(Thread.MAX_PRIORITY);
                return thread;
            });
    
    
            // Record the start time for measuring script execution duration
            long start = System.currentTimeMillis();
    
            // Submit the Kappalab R script execution for asynchronous processing
            Future<KappalabOutput> res = executor.submit(kappalabRScript);
    
            try {
                // Retrieve the KappalabOutput
                KappalabOutput output = timeLimit == 0 ? res.get() : res.get(timeRemaining, TimeUnit.MILLISECONDS);
    
                // Measure the time taken for script execution
                long time = System.currentTimeMillis() - start;
                timeRemaining -= time;
    
                // If there are error messages in the output, log them
                if (output.getErrorMessages() != null) {
                    return FunctionUtil.logErrorFunction(output.getErrorMessages());
                }
    
                // Return learned capacities and time taken
                return FunctionUtil.getFunctionParameters(ChoquetMobiusScoreFunction.TYPE, nbMeasures, kAdditivity,
                        output.getCapacities(), time / 1000d);
            } catch (TimeoutException e) {
                String[] errorMessages = { "Timeout while waiting for Kappalab R script to finish." };
                return FunctionUtil.logErrorFunction(errorMessages);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                String[] errorMessages = { "Execution was interrupted." };
                return FunctionUtil.logErrorFunction(errorMessages);
            } catch (ExecutionException | CancellationException e) {
                // Handle execution exceptions from the Future
                String causeMessage = e.getCause() != null ? e.getCause().getMessage() : "Unknown error";
                System.err.println("Error in Kappalab: " + causeMessage);
    
                // Remove the last added alternative and retry
                if (!rankingsCopy.isEmpty()) {
                    System.err.println("Removing last added alternative and retrying.");
                    rankingsCopy.remove(rankingsCopy.size() - 1);
                    // Continue the loop to retry with fewer rankings
                } else {
                    // No more alternatives to remove
                    System.err.println("No more alternatives to remove.");
                    String[] errorMessages = { "Failed to learn from rankings after removing all alternatives.", causeMessage };
                    return FunctionUtil.logErrorFunction(errorMessages);
                }
            } finally {
                executor.shutdown();
            }
        }
    
        // If we exit the loop, it means we couldn't succeed
        System.err.println("Failed to learn from rankings. All alternatives have been removed.");
        String[] errorMessages = { "Failed to learn from rankings. All alternatives have been removed." };
        return FunctionUtil.logErrorFunction(errorMessages);
    }    
}
