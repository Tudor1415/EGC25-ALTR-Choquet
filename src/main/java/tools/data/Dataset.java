package tools.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.zaxxer.sparsebits.SparseBitSet;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.utils.RandomUtil;

/**
 * This class encodes a transactional dataset (ex: from a dat file) into an
 * array of items
 * 
 * @param filename        The name of the file containing the transactional
 *                        dataset
 * @param expDir          The name of the directory in which the data is
 *                        situated
 * @param classItemValues A sequence of values that the class items can take
 * @throws IOException If the specified data file doesn't exist
 */
@Getter
public class Dataset {

    // Variables regarding the path to the file containing the transactional data
    private String filename;
    private String expDir;

    // Variables regarding the dataset
    private @Setter @Getter Map<String, SparseBitSet> itemsMap; // The map value -> item coverage in the dataset
    private @Setter @Getter Set<String> consequentItemsSet; // The set of the values all the class items
    private @Setter @Getter Set<String> antecedentItemsSet; // The set of the values of all the antecedent items
    private @Setter @Getter String[] consequentItemsArray; // The array of the values all the class items
    private @Setter @Getter String[] antecedentItemsArray; // The array of the values of all the antecedent items
    private @Setter @Getter int nbAntecedentItems; // The number of different antecedent items
    private @Setter @Getter int nbConsequentItems; // The number of different consequent items
    private @Setter @Getter int nbTransactions; // The number of transactions in the transactional dataset
    private @Setter @Getter String[][] transactions; // The array of transactions red from the dat file.
    private @Getter UnionFind equivalenceClasses; // The item equivalence classes

    public Dataset(String filename, String expDir, Set<String> consequentItemsSet) throws IOException {
        // Variables regarding the file
        this.filename = filename;
        this.expDir = expDir;

        // Initializing variables regarding the data
        this.transactions = getTransactionalDataset();
        this.nbTransactions = this.transactions.length;
        getItemsFromTransactions();

        // Initializing consequent and antecedent items
        setConsequentItemsSet(consequentItemsSet);
        setConsequentItemsArray(getConsequentItemsSet().toArray(new String[0]));
        initializeAntecedentItemsValues();

        // Initializing the number of items for sampling
        this.nbAntecedentItems = getAntecedentItemsSet().size();
        this.nbConsequentItems = getConsequentItemsSet().size();

        // Find the equivalence classes
        findEquivalenceClasses();
    }

    public Dataset(String[][] transactionalDataset, Set<String> consequentItemsSet) {
        this.transactions = transactionalDataset;
        this.nbTransactions = transactionalDataset.length;
        getItemsFromTransactions();

        // Initializing class and antecedent items
        setConsequentItemsSet(consequentItemsSet);
        setConsequentItemsArray(getConsequentItemsSet().toArray(new String[0]));
        initializeAntecedentItemsValues();

        // Initializing the number of items for sampling
        this.nbAntecedentItems = getAntecedentItemsSet().size();
        this.nbConsequentItems = getConsequentItemsSet().size();

        // Find the equivalence classes
        findEquivalenceClasses();
    }

    public void findEquivalenceClasses() {
        HashSet<String> neverSeen = new HashSet<>();
        neverSeen.addAll(getConsequentItemsSet());
        neverSeen.addAll(getAntecedentItemsSet());

        this.equivalenceClasses = new UnionFind(neverSeen.toArray(new String[0]));

        String[][] transactions = getTransactions();

        while (!neverSeen.isEmpty()) {
            for (String[] transaction : transactions) {
                String classRep = transaction[0];

                for (String item : transaction) {
                    equivalenceClasses.union(item, classRep);
                    neverSeen.remove(item);
                }
            }
        }
    }

    /**
     * Initializes the set of antecedent items values as all the possible item
     * values except the ones in the array of class items values.
     * The antecedent items are the items that are not part of the class items.
     */
    private void initializeAntecedentItemsValues() {
        // Filter out items that are part of consequent items
        setAntecedentItemsSet(getItemsMap().keySet().stream()
                .filter(key -> !getConsequentItemsSet().contains(key))
                .collect(Collectors.toSet()));
        setAntecedentItemsArray(getAntecedentItemsSet().toArray(new String[0]));
    }

    /**
     * This method is used to retrieve the transactional data from a dat file.
     *
     * @param filename The name of the file in the data directory.
     * @return A list of arrays representing the transactional data.
     * @throws IOException If there is an issue reading the file.
     */
    public String[][] getTransactionalDataset() throws IOException {
        List<String[]> transactions = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(expDir + filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split("\\s+");
                String[] transaction = new String[values.length];

                // Parse each value and add it to the transaction array
                for (int i = 0; i < values.length; i++) {
                    transaction[i] = values[i];
                }

                // Add the transaction to the list
                transactions.add(transaction);
            }
        }

        // Convert List<String[]> to String[][]
        String[][] transactionArray = new String[transactions.size()][];
        for (int i = 0; i < transactions.size(); i++) {
            transactionArray[i] = transactions.get(i);
        }

        return transactionArray;
    }

    /**
     * Converts the transactional dataset into a set of unique items.
     *
     * @param transactions The transactional dataset.
     * @return A set containing all the unique items from the transactional dataset.
     */
    public void getItemsFromTransactions() {
        this.itemsMap = new HashMap<>();

        // Iterate through transactions to identify unique items and their occurrences
        for (int transactionIndex = 0; transactionIndex < transactions.length; transactionIndex++) {
            String[] transaction = this.transactions[transactionIndex];
            for (int itemIndex = 0; itemIndex < transaction.length; itemIndex++) {
                String itemValue = String.valueOf(transaction[itemIndex]);
                itemsMap.computeIfAbsent(itemValue, k -> new SparseBitSet()).set(transactionIndex);
            }
        }
    }

    /**
     * Returns a list of size nbRules of rules extracted from random transactions
     * in the transactional dataset.
     * 
     * @param nbRules The desired number of rules.
     * @return A list of random rules.
     */
    public List<DecisionRule> getRandomValidRules(int nbRules, double smoothCounts, String[] measureNames) {
        /* Initializing the random instance */
        RandomUtil random = new RandomUtil();

        int nbTransactions = this.getTransactions().length;

        List<DecisionRule> rules = new ArrayList<>();

        /* Selecting random transactions and converting them to rules */
        for (int i = 0; i < nbRules; i++) {
            int transactionIndex = random.nextInt(nbTransactions);
            String[] transaction = this.getTransactions()[transactionIndex];

            List<String> shuffledItems = new ArrayList<>(Arrays.asList(transaction));
            Collections.shuffle(shuffledItems);

            /* Creating a new rule and adding each item from the transaction to it */
            DecisionRule selectedDecisionRule = new DecisionRule(new HashSet<>(), "", this, 100, 100, smoothCounts,
                    measureNames);

            for (String item : shuffledItems) {
                /* Adding each item either to the consequent or to the antecedent of the rule */
                if (getAntecedentItemsSet().contains(item)) {
                    if (random.Bernoulli(0.5))
                        selectedDecisionRule.addToX(item);
                } else {
                    selectedDecisionRule.setY(item);
                }
            }

            /* Adding the decision rule in to the final list */
            rules.add(selectedDecisionRule);
        }

        return rules;
    }
}
