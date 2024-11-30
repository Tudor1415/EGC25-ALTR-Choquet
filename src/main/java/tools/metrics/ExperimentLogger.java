package tools.metrics;

import java.io.File;
import java.util.Set;
import java.util.List;
import java.time.ZoneId;
import java.time.Instant;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.time.LocalDateTime;
import java.util.stream.Collectors;
import java.beans.PropertyChangeEvent;
import tools.normalization.Normalizer;
import java.beans.PropertyChangeListener;
import java.time.format.DateTimeFormatter;

import tools.rules.DecisionRule;
import tools.alternatives.Alternative;
import tools.oracles.ArtificialOracle;
import tools.alternatives.IAlternative;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;

/**
 * ExperimentLogger modified to compute and log the time it takes to finish
 * each learning iteration.
 */
public class ExperimentLogger implements PropertyChangeListener {

    private ArtificialOracle oracle;
    private String learningAlgName, loggingPath, datasetName;
    private int foldIdx;
    private List<DecisionRule> testRuleSet;
    private NormalizationMethod normMethod;
    private Normalizer normalizer;
    private int iteration = 0;

    // Variables to store iteration times
    private List<Long> perIterationTimes = new ArrayList<>();
    private long lastUpdateTime;

    public ExperimentLogger(ArtificialOracle oracle, String learningAlgName, String loggingPath, String datasetName,
            int foldIdx, List<DecisionRule> testRuleSet, NormalizationMethod normMethod) {
        this.oracle = oracle;
        this.learningAlgName = learningAlgName;
        this.loggingPath = loggingPath;
        this.datasetName = datasetName;
        this.foldIdx = foldIdx;
        this.testRuleSet = testRuleSet;
        this.normMethod = normMethod;

        File logDir = new File(loggingPath);
        if (!logDir.exists()) {
            boolean created = logDir.mkdirs();
            if (created) {
                System.out.println("Logging path created: " + loggingPath);
            } else {
                System.err.println("Failed to create logging path: " + loggingPath);
            }
        }

        this.normalizer = new Normalizer();
        initNormalization();

        // Initialize lastUpdateTime
        this.lastUpdateTime = System.currentTimeMillis();
    }

    private void initNormalization() {
        for (DecisionRule rule : testRuleSet)
            this.normalizer.normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);
    }

    /**
     * Writes a list of alternatives to a CSV file.
     *
     * @param rules        The list of decision rules to write to the CSV file.
     * @param scoresApprox The list of approximate scores associated with each rule.
     * @param scoresOracle The list of oracle scores associated with each rule.
     * @param fileName     The name of the file to write.
     */
    public void writeSampleToCSV(List<DecisionRule> rules, List<Double> scoresApprox, List<Double> scoresOracle,
            String fileName) {
        Instant now = Instant.now();
        String timestamp = DateTimeFormatter.ofPattern("yyyyMMddHHmmss")
                .withZone(ZoneId.systemDefault())
                .format(now) + String.format("%09d", now.getNano());
        String filePath = loggingPath + "/" + fileName + "_" + timestamp + ".csv";

        try (FileWriter writer = new FileWriter(filePath)) {
            // Write header
            if (iteration == 0) {
                writer.append("Rule,");
                for (int i = 0; i < rules.get(0).getAlternative().getVector().length; i++) {
                    writer.append(rules.get(0).getMeasureNames()[i] + ",");
                }
            }

            writer.append("scoreApprox,");
            writer.append("scoreOracle,");
            writer.append("\n");

            // Write data
            for (int i = 0; i < rules.size(); i++) {
                if (iteration == 0) {
                    writer.append(ruleToString(rules.get(i)) + ",");
                    double[] vector = rules.get(i).getAlternative().getVector();
                    for (double value : vector) {
                        writer.append(Double.toString(value)).append(",");
                    }
                }
                writer.append(scoresApprox.get(i) + ",");
                writer.append(scoresOracle.get(i) + ",");
                writer.append("\n");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Listens for property change events and computes metrics when the score
     * function changes.
     *
     * @param evt The property change event.
     */
    @Override
    public void propertyChange(PropertyChangeEvent evt) {
        // Record the current time and compute the time since the last update
        long currentTime = System.currentTimeMillis();
        long iterationTime = currentTime - lastUpdateTime;
        perIterationTimes.add(iterationTime);
        lastUpdateTime = currentTime;

        // Retrieve the last available approximation function.
        ISinglevariateFunction func = (ISinglevariateFunction) evt.getNewValue();

        String filename = datasetName + "_" + foldIdx + "_"
                + learningAlgName + "_" + this.oracle.getScoreFunction().getName();

        List<Double> approxScore = testRuleSet.parallelStream()
                .map(rule -> getValidRuleScore(rule, func))
                .collect(Collectors.toList());

        List<Double> oracleScore = testRuleSet.parallelStream()
                .map(rule -> oracle.computeScore(rule))
                .collect(Collectors.toList());

        writeSampleToCSV(testRuleSet, approxScore, oracleScore, filename);
        iteration += 1;
    }

    private double getValidRuleScore(DecisionRule rule, ISinglevariateFunction scoreFunction) {
        double[] unNormVector = rule.getAlternative().getVector();
        double[] normVector = this.normalizer.normalize(unNormVector, this.normMethod, false);

        IAlternative normAlternative = new Alternative(normVector);
        return scoreFunction.computeScore(normAlternative, rule);
    }

    private String ruleToString(DecisionRule rule) {
        Set<String> antecedentValues = rule.getItemsInX();
        String consequentValues = rule.getY();

        return "[" + String.join("; ", antecedentValues) + "]" + " => " + "[" + String.join("; ", consequentValues)
                + "]";
    }

    /**
     * Writes the iteration times to a CSV file.
     */

    public void writeIterationTimes(String oracleName) {
        String directoryPath = loggingPath;

        if (loggingPath.endsWith("/samples/")) {
            directoryPath = loggingPath.substring(0, loggingPath.length() - "/samples/".length());
        }

        directoryPath = directoryPath + "/timing/";
        String filename = directoryPath + learningAlgName + "_" + oracleName + "_times_fold_" + foldIdx + ".csv";

        // Ensure the directory exists
        File directory = new File(directoryPath);
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                System.err.println("Failed to create directory: " + directoryPath);
                return;
            }
        }

        // Write the file
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Iteration,Time(ms)\n");
            for (int i = 0; i < perIterationTimes.size(); i++) {
                writer.write((i + 1) + "," + perIterationTimes.get(i) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
