package tools.metrics;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import tools.normalization.Normalizer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import tools.alternatives.Alternative;
import tools.alternatives.IAlternative;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.oracles.ArtificialOracle;
import tools.rules.DecisionRule;

/**
 * Dashboard for computing ranking metrics based on predicted rankings compared
 * to a reference ranking.
 * It listens for property change events and computes metrics accordingly.
 */
public class ExperimentLogger implements PropertyChangeListener {

    private ArtificialOracle oracle;
    private String learningAlgName, loggingPath, datasetName;
    private int foldIdx;
    private List<DecisionRule> testRuleSet;
    private NormalizationMethod normMethod;
    private Normalizer normalizer;
    private int iteration = 0;

    public ExperimentLogger(ArtificialOracle oracle, String learningAlgName, String loggingPath, String datasetName,
            int foldIdx,
            List<DecisionRule> testRuleSet, NormalizationMethod normMethod) {
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
        } else {
            // System.out.println("Logging path already exists: " + loggingPath);
        }

        this.normalizer = new Normalizer();
        initNormalization();
    }

    private void initNormalization() {
        for (DecisionRule rule : testRuleSet)
            this.normalizer.normalize(rule.getAlternative().getVector(), NormalizationMethod.NO_NORMALIZATION, true);

    }

    /**
     * Writes a list of alternatives to a CSV file.
     *
     * @param alternatives The list of alternatives to write to the CSV file.
     * @param scoresList   The list of scores associated with each alternative.
     */
    public void writeSampleToCSV(List<DecisionRule> rules, List<Double> scoresApprox, List<Double> scoresOracle,
            String fileName) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String filePath = loggingPath + fileName + "_" + timestamp + ".csv";

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

            // System.out.println("CSV file has been created successfully!");

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
}
