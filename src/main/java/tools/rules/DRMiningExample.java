package tools.rules;

import java.io.FileWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;

import io.gitlab.chaver.mining.patterns.constraints.factory.ConstraintFactory;
import io.gitlab.chaver.mining.patterns.io.DatReader;
import io.gitlab.chaver.mining.patterns.io.TransactionalDatabase;

public class DRMiningExample {

    static int[] getItemset(BoolVar[] x, TransactionalDatabase database) {
        return IntStream
                .range(0, x.length)
                .filter(i -> x[i].getValue() == 1)
                .map(i -> database.getItems()[i])
                .toArray();
    }

    private static Set<Integer> getClassItems(String datasetName) {
        switch (datasetName) {
            case "adult":
                return new HashSet<>(Arrays.asList(145, 146));
            case "bank":
                return new HashSet<>(Arrays.asList(89, 90));
            case "connect":
                return new HashSet<>(Arrays.asList(127, 128));
            case "credit":
                return new HashSet<>(Arrays.asList(111, 112));
            case "dota":
                return new HashSet<>(Arrays.asList(346, 347));
            case "toms":
                return new HashSet<>(Arrays.asList(911, 912));
            case "mushroom":
                return new HashSet<>(Arrays.asList(116, 117));
            default:
                return null;
        }
    }

    static void consequentItemsConstraint(Model model, BoolVar[] y, Set<Integer> classItems,
            TransactionalDatabase database) {
        Map<Integer, Integer> itemMap = database.getItemsMap();
        BoolVar[] selected = classItems.stream().map(i -> y[itemMap.get(i)]).toArray(BoolVar[]::new);
        model.or(selected).post();
        model.sum(y, "=", 1).post();
    }

    public static void mine(String dataPath, Set<Integer> classItems, String outputCsvPath) throws Exception {
        TransactionalDatabase database = new DatReader(dataPath).read();
        // Min frequency of the rule (absolute value)
        int minFreq = 10;
        // Min confidence of the rule (percentage)
        int minConf = 90;
        Model model = new Model("Association Rule mining");
        // Antecedent of the rule : x
        BoolVar[] x = model.boolVarArray("x", database.getNbItems());
        // Consequent of the rule : y
        BoolVar[] y = model.boolVarArray("y", database.getNbItems());
        // z = x U y
        BoolVar[] z = model.boolVarArray("z", database.getNbItems());
        for (int i = 0; i < database.getNbItems(); i++) {
            // Ensure that an item i is not in the antecedent and consequent of the rule at
            // the same time
            model.arithm(x[i], "+", y[i], "<=", 1).post();
            // z[i] = x[i] OR y[i]
            model.addClausesBoolOrEqVar(x[i], y[i], z[i]);
        }
        // sum(x) >= 1 (i.e. the antecedent of the rule is not empty)
        model.addClausesBoolOrArrayEqualTrue(x);
        // sum(y) >= 1 (i.e. the consequent of the rule is not empty)
        model.addClausesBoolOrArrayEqualTrue(y);
        // Frequency of z
        IntVar freqZ = model.intVar("freqZ", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqZ, z).post();
        // Frequency of x
        IntVar freqX = model.intVar("freqX", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqX, x).post();

        // Frequency of y
        IntVar freqY = model.intVar("freqY", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqY, y).post();

        // Confidence of the rule = freqZ / freqX (multiplied by 100 to get an integer
        // variable)
        freqZ.mul(100).ge(freqX.mul(minConf)).post();
        // Only class items in the consequence of the rule
        consequentItemsConstraint(model, y, classItems, database);

        // Write the results to a CSV file, limiting to a maximum of 10,000 rules
        try (Writer writer = new FileWriter(outputCsvPath)) {
            writer.write("antecedent,consequent,freqX,freqY,freqZ\n");

            Solver solver = model.getSolver();
            int ruleCount = 0;  // Counter for the number of rules written
            while (solver.solve() && ruleCount < 10000) {
                int[] antecedent = getItemset(x, database);
                int[] consequent = getItemset(y, database);

                // Write rule data to the CSV
                writer.write("{"
                        + Arrays.stream(antecedent).mapToObj(String::valueOf).reduce((a, b) -> a + ";" + b).orElse("")
                        + "}," +
                        "{"
                        + Arrays.stream(consequent).mapToObj(String::valueOf).reduce((a, b) -> a + ";" + b).orElse("")
                        + "}," +
                        freqX.getValue() + "," +
                        freqY.getValue() + "," +
                        freqZ.getValue() + "\n");

                ruleCount++;  // Increment the rule counter
            }

            solver.printStatistics();
        }
    }

    public static void main(String[] args) throws Exception {
        for (int i = 1; i <= 10; i++) {
            System.out.println("Fold: " + i);
            String dataPath = "data/folds/toms/train/train_" + i + ".dat";
            Set<Integer> classItems = getClassItems("toms");
            String outputCsvPath = "data/folds/toms/train/train_rules_" + i + ".csv";
            mine(dataPath, classItems, outputCsvPath);
        }
    }
}
