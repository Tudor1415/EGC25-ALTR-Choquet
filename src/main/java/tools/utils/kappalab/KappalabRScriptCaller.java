package tools.utils.kappalab;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.concurrent.Callable;

import com.google.gson.Gson;

import lombok.AllArgsConstructor;

/**
 * Callable task for invoking Kappalab R script to compute Choquet integral
 * values.
 * This class is used to asynchronously call the Kappalab R script with
 * specified input
 * and obtain the computed output.
 */
@AllArgsConstructor
public class KappalabRScriptCaller implements Callable<KappalabOutput> {

    /** The input file for Kappalab, containing parameters and preferences. */
    private File inputFile;

    /** The output file to store the result computed by Kappalab. */
    private File outputFile;

    /** The input parameters for Kappalab Choquet integral. */
    private KappalabInput input;

    /**
     * Executes the Kappalab R script with the provided input and returns the
     * computed output.
     *
     * @return The computed Choquet integral values.
     * @throws Exception If an error occurs during the execution or processing of
     *                   the R script.
     */
    @Override
    public KappalabOutput call() throws Exception {
        // Serialize input to JSON and write to the input file
        Gson gson = new Gson();
        BufferedWriter writer = new BufferedWriter(new FileWriter(inputFile));
        writer.write(gson.toJson(input));
        writer.close();

        // Execute the Kappalab R script using ProcessBuilder
        // "C:\Program Files\R\R-4.3.2\bin\x64\Rscript.exe"
        ProcessBuilder builder = new ProcessBuilder("Rscript",
                "scripts/call_kappalab.R",
                inputFile.getAbsolutePath(), outputFile.getAbsolutePath());
        Process process = builder.start();
        process.waitFor();

        // Check if the R script executed successfully
        if (process.exitValue() != 0) {
            throw new RuntimeException("Error in R script");
        }

        // Deserialize the output from the result file
        return gson.fromJson(new FileReader(outputFile), KappalabOutput.class);
    }
}
