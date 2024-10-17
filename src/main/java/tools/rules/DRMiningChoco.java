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

public class DRMiningChoco {

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
            case "iris":
                return new HashSet<>(Arrays.asList(12, 13));
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

    public static void mine(String dataPath, Set<Integer> classItems, String outputCsvPath, int minFreq, int minConf)
            throws Exception {
        TransactionalDatabase database = new DatReader(dataPath).read();
        Model model = new Model("Association Rule mining");
        BoolVar[] x = model.boolVarArray("x", database.getNbItems());
        BoolVar[] y = model.boolVarArray("y", database.getNbItems());
        BoolVar[] z = model.boolVarArray("z", database.getNbItems());
        for (int i = 0; i < database.getNbItems(); i++) {
            model.arithm(x[i], "+", y[i], "<=", 1).post();
            model.addClausesBoolOrEqVar(x[i], y[i], z[i]);
        }
        model.addClausesBoolOrArrayEqualTrue(x);
        model.addClausesBoolOrArrayEqualTrue(y);

        IntVar freqZ = model.intVar("freqZ", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqZ, z).post();
        IntVar freqX = model.intVar("freqX", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqX, x).post();
        IntVar freqY = model.intVar("freqY", minFreq, database.getNbTransactions());
        ConstraintFactory.coverSize(database, freqY, y).post();
        freqZ.mul(100).ge(freqX.mul(minConf)).post();
        consequentItemsConstraint(model, y, classItems, database);

        try (Writer writer = new FileWriter(outputCsvPath)) {
            writer.write("antecedent,consequent,freqX,freqY,freqZ\n");
            Solver solver = model.getSolver();
            int ruleCount = 0;
            while (solver.solve() && ruleCount < 10000) {
                int[] antecedent = getItemset(x, database);
                int[] consequent = getItemset(y, database);
                writer.write("{"
                        + Arrays.stream(antecedent).mapToObj(String::valueOf).reduce((a, b) -> a + ";" + b).orElse("")
                        + "}," +
                        "{"
                        + Arrays.stream(consequent).mapToObj(String::valueOf).reduce((a, b) -> a + ";" + b).orElse("")
                        + "}," +
                        freqX.getValue() + "," +
                        freqY.getValue() + "," +
                        freqZ.getValue() + "\n");

                ruleCount++;
            }
            solver.printStatistics();
        }
    }

    public static void main(String[] args) throws Exception {
        String dataPath = args.length > 0 ? args[0] : "data/dat-files/iris.dat";
        String datasetName = args.length > 1 ? args[1] : "iris";
        String outputPath = args.length > 2 ? args[2] : "data/mined_rules/iris.csv";
        int minSupport = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int minConfidence = args.length > 4 ? Integer.parseInt(args[4]) : 1;

        Set<Integer> classItems = getClassItems(datasetName);
        if (classItems == null) {
            System.out.println("Invalid dataset name provided.");
            return;
        }

        mine(dataPath, classItems, outputPath, minSupport, minConfidence);
    }
}
