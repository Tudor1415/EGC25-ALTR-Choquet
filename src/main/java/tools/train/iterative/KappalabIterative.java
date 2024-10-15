package tools.train.iterative;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import lombok.Setter;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.FunctionParameters;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.Choquet.ChoquetMobiusScoreFunction;
import tools.ranking.Ranking;
import tools.ranking.RankingsProvider;
import tools.train.IterativeRankingLearn;
import tools.utils.FunctionUtil;
import tools.utils.kappalab.KappalabInput;
import tools.utils.kappalab.KappalabOutput;
import tools.utils.kappalab.KappalabRScriptCaller;
import tools.utils.kappalab.KappalabUtils;

/**
 * Kappalab Iterative is a learning class that communicates with an R script
 * using Json files.
 *
 * @param nbIterations     Number of iterations for the iterative learning
 *                         process.
 * @param rankingsProvider The provider of rankings used for learning (e.g. the
 *                         heuristic).
 * @param func             The score function representing the current state of
 *                         learning (e.g. the Choquet integral).
 * @param nbMeasures       Number of measures/criteria in the ranking.
 */
public class KappalabIterative extends IterativeRankingLearn {

    @Setter
    private double delta = 1e-6d;
    @Setter
    private int kAdditivity = 2;
    @Setter
    private String approachType = "Generalized Least Squares";

    public KappalabIterative(int nbIterations, RankingsProvider rankingsProvider, ISinglevariateFunction func,
            int nbMeasures) {
        super(nbIterations, rankingsProvider, func, nbMeasures);
    }

    /**
     * Learns the ranking function iteratively using the Kappalab R script to train
     * Choquet integral.
     *
     * @param rankings The list of rankings used for learning.
     * @return FunctionParameters representing the learned ranking function.
     * @throws Exception If an error occurs during the learning process.
     */
    @Override
    public FunctionParameters learnFromRankings(List<Ranking<IAlternative>> rankings) throws Exception {
        KappalabInput input = new KappalabInput(kAdditivity, approachType);

        // Add each ranking to the Kappalab input.
        for (Ranking<IAlternative> ranking : rankings) {
            KappalabUtils.addRankingToKappalabInput(ranking, input, delta);
        }

        // Create temporary files to store Kappalab input and output data.
        File inputFile = File.createTempFile("kappalab_input", ".json");
        File outputFile = File.createTempFile("kappalab_output", ".json");

        // We use this to call the R script.
        KappalabRScriptCaller kappalabRScript = new KappalabRScriptCaller(inputFile, outputFile, input);

        // Create a single-threaded executor for running the Kappalab R script.
        ExecutorService executor = Executors.newSingleThreadExecutor();

        // Record the start time for measuring script execution duration.
        long start = System.currentTimeMillis();

        // Submit the Kappalab R script execution for asynchronous processing.
        Future<KappalabOutput> res = executor.submit(kappalabRScript);

        try {
            // Retrieve the KappalabOutput.
            KappalabOutput output = timeLimit == 0 ? res.get() : res.get(timeRemaining, TimeUnit.MILLISECONDS);

            // Measure the time taken for script execution.
            long time = System.currentTimeMillis() - start;
            timeRemaining -= time;

            // If there are error messages in the output, log them.
            if (output.getErrorMessages() != null) {
                return FunctionUtil.logErrorFunction(output.getErrorMessages());
            }

            // Return learned capacities and time taken.
            return FunctionUtil.getFunctionParameters(ChoquetMobiusScoreFunction.TYPE, nbMeasures, kAdditivity,
                    output.getCapacities(), time / 1000d);
        } catch (TimeoutException e) {
            return timeOut();
        } finally {
            executor.shutdown();
        }
    }
}
