package tools.oracles;

import java.util.Arrays;
import java.util.Scanner;

import tools.rules.DecisionRule;

public class TicTacToeOracle implements Oracle {
    public String TYPE = "TicTacToe - Human";

    private static final Scanner scanner = new Scanner(System.in);

    /**
     * Compares two rules based on human preference.
     *
     * @param rule_a The first rule to compare.
     * @param rule_b The second rule to compare.
     * @return A negative integer, zero, or a positive integer if the score of
     *         rule 'a' is less than, equal to, or greater than the score of
     *         rule 'b', respectively.
     */
    @Override
    public int compare(DecisionRule rule_a, DecisionRule rule_b) {
        showVisualization(rule_a, rule_b);
    
        System.out.println("Which rule do you prefer?");
        System.out.println("If the rule name is in green, it means that the X player wins.");
        System.out.println("Enter 'a' if you prefer Rule A over Rule B, or 'b' if you prefer Rule B over Rule A: ");
    
        String choice = "";
        while (!choice.equals("a") && !choice.equals("b")) {
            try {
                choice = scanner.nextLine().trim().toLowerCase();
                if (!choice.equals("a") && !choice.equals("b")) {
                    System.out.println("Please enter 'a' or 'b'.");
                }
            } catch (Exception e) {
                System.out.println("Invalid input. Please enter 'a' or 'b'.");
            }
        }
    
        return (choice.equals("a")) ? 1 : -1;
    }

    /**
     * Provides a rule pair visualization method.
     * The rules are displayed side by side using the tic-tac-toe map.
     * - If the rule's class item (rule.getY()) is 28, the board will have a green
     * border.
     * - If the rule's class item (rule.getY()) is 29, the board will have a red
     * border.
     * 
     * @param rule_a First decision rule to visualize.
     * @param rule_b Second decision rule to visualize.
     */
    public void showVisualization(DecisionRule rule_a, DecisionRule rule_b) {
        // Define the color codes for the class items
        String border_a = rule_a.getY().equals("28") ? "\u001B[32m" : "\u001B[31m"; // Green or Red for class
        String border_b = rule_b.getY().equals("28") ? "\u001B[32m" : "\u001B[31m"; // Green or Red for class
        String resetColor = "\u001B[0m"; // Reset color

        // Get the tic-tac-toe grids for both rules
        String[] grid_a = getTicTacToeGrid(rule_a);
        String[] grid_b = getTicTacToeGrid(rule_b);

        // Display the boards side by side with 5 spaces separating the middle line
        String separator = "     |     "; // 5 spaces on each side of the vertical separator

        System.out.println(border_a + (rule_a.getY().equals("28") ? "X Wins!" : "O Wins!") + resetColor + "       |     " + border_b + (rule_b.getY().equals("28") ? "X Wins!" : "O Wins!") + resetColor);
        for (int i = 0; i < 3; i++) {
            System.out.println(grid_a[i] + separator + grid_b[i]);
        }
    }

    /**
     * Converts the rule's antecedents and class into a tic-tac-toe grid format.
     * 
     * @param rule The decision rule whose grid is to be generated.
     * @return A 3x3 grid representing the rule's antecedent.
     */
    private String[] getTicTacToeGrid(DecisionRule rule) {
        String[] grid = new String[3];

        // Create the grid based on the items in the rule's antecedent
        String[] map = new String[9];
        Arrays.fill(map, " ");
        
        for (String antecedent : rule.getItemsInX()) {
            int antecedentValue = Integer.parseInt(antecedent);
            int mapIndex = (antecedentValue - 1) % 9;
        
            // Determine what to place in the map based on the value
            if (antecedentValue % 3 == 0) {
                map[mapIndex] = "x";
            } else if (antecedentValue % 3 == 1) {
                map[mapIndex] = "b";
            } else if (antecedentValue % 3 == 2) {
                map[mapIndex] = "o";
            }
        }
        
        // Fill in the rows with the corresponding tic-tac-toe values
        grid[0] = map[0] + " | " + map[1] + " | " + map[2];
        grid[1] = map[3] + " | " + map[4] + " | " + map[5];
        grid[2] = map[6] + " | " + map[7] + " | " + map[8];

        return grid;
    }
}
