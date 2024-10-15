package tools.alternatives;

public interface IAlternative {

    double[] getVector();

    default boolean equals(IAlternative other) {
        if (other == null) {
            return false;
        }
        double[] thisVector = this.getVector();
        double[] otherVector = other.getVector();
        if (thisVector.length != otherVector.length) {
            return false;
        }
        for (int i = 0; i < thisVector.length; i++) {
            if (thisVector[i] != otherVector[i]) {
                return false;
            }
        }
        return true;
    }

    double getOrderedValue(int i);

    int[] getOrderedPermutation();

    IAlternative deepCopy();
}
