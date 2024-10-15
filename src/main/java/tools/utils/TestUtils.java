package tools.utils;

public class TestUtils {
    /**
     * Counts the number of transactions in the given array that contain all of the
     * specified items.
     *
     * @param transactions the array of transactions to search through
     * @param items        the items to search for in each transaction
     * @return the number of transactions that contain all of the specified items
     */
    public static int countTransactionsWithItems(String[][] transactions, String... items) {
        int count = 0;

        // iterate over each transaction in the array
        for (String[] transaction : transactions) {
            boolean containsAllItems = true;

            // iterate over each item in the list of items to search for
            for (String item : items) {
                boolean containsItem = false;

                // iterate over each item in the transaction
                for (String t : transaction) {
                    if (t.equals(item)) {
                        containsItem = true;
                        break;
                    }
                }

                if (!containsItem) {
                    containsAllItems = false;
                    break;
                }
            }

            if (containsAllItems) {
                count++;
            }
        }

        return count;
    }
}
