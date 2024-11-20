package tools.utils;
import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;

public class AlternativeScore implements Comparable<AlternativeScore> {
    @Getter @Setter
    private IAlternative alternative;
    @Getter @Setter
    private double score;

    public AlternativeScore(IAlternative alternative, double score) {
        this.alternative = alternative;
        this.score = score;
    }

    @Override
    public int compareTo(AlternativeScore other) {
        int compare = Double.compare(this.score, other.score);
        if (compare != 0) {
            return compare;
        } else {
            // Ensure consistent ordering for equal scores
            return Integer.compare(System.identityHashCode(this.alternative), System.identityHashCode(other.alternative));
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof AlternativeScore) {
            AlternativeScore other = (AlternativeScore) obj;
            return this.score == other.score && this.alternative.equals(other.alternative);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(score) * 31 + alternative.hashCode();
    }
}
