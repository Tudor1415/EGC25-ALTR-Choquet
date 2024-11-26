package tools.ranking.heuristics;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
class DominationVector {
    private final int[] vector;

    public DominationVector(int[] vector) {
        this.vector = vector;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        DominationVector that = (DominationVector) o;
        return Arrays.equals(vector, that.vector);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(vector);
    }

    @Override
    public String toString() {
        return Arrays.toString(vector);
    }
}
