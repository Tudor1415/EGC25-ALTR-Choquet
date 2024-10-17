package tools.data;

import java.util.HashMap;
import java.util.Map;

public class UnionFind {
    private Map<String, Integer> indexMap; // Map to hold the string to index mapping
    private int[] parent;
    private int[] rank;

    public UnionFind(String[] elements) {
        indexMap = new HashMap<>();
        parent = new int[elements.length];
        rank = new int[elements.length];
        for (int i = 0; i < elements.length; i++) {
            indexMap.put(elements[i], i);
            parent[i] = i;
            rank[i] = 0;
        }
    }

    // Find method using String element
    public String find(String element) {
        if (!indexMap.containsKey(element)) {
            throw new IllegalArgumentException("Element not found in union-find");
        }
        int index = indexMap.get(element);
        int rootIndex = find(index);
        return getKeyByValue(rootIndex);
    }

    private int find(int index) {
        if (parent[index] != index) {
            parent[index] = find(parent[index]);
        }
        return parent[index];
    }

    public void union(String element1, String element2) {
        if (!indexMap.containsKey(element1) || !indexMap.containsKey(element2)) {
            throw new IllegalArgumentException("One or both elements not found in union-find");
        }
        int index1 = indexMap.get(element1);
        int index2 = indexMap.get(element2);

        int rootP = find(index1);
        int rootQ = find(index2);
        if (rootP != rootQ) {
            // Union by rank
            if (rank[rootP] > rank[rootQ]) {
                parent[rootQ] = rootP;
            } else if (rank[rootP] < rank[rootQ]) {
                parent[rootP] = rootQ;
            } else {
                parent[rootQ] = rootP;
                rank[rootP]++;
            }
        }
    }

    private String getKeyByValue(int value) {
        for (Map.Entry<String, Integer> entry : indexMap.entrySet()) {
            if (entry.getValue().equals(value)) {
                return entry.getKey();
            }
        }
        return null;
    }

    // Method to count the number of distinct equivalence classes
    public int countClasses() {
        int count = 0;
        for (int i = 0; i < parent.length; i++) {
            if (i == find(i)) {
                count++;
            }
        }
        return count;
    }
}
