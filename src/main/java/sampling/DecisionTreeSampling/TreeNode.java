package sampling.DecisionTreeSampling;

import java.util.Set;

public class TreeNode {
    public Set<String> itemset;
    public TreeNode includedChild;
    public TreeNode excludedChild;
    public String classLabel;

    public TreeNode(Set<String> itemset, String classLabel) {
        this.itemset = itemset;
        this.includedChild = null;
        this.excludedChild = null;
        this.classLabel = classLabel;
    }

    public boolean isLeaf() {
        return includedChild == null && excludedChild == null;
    }
}
