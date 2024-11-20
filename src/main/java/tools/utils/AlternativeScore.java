package tools.utils;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;

public class AlternativeScore implements Comparable<AlternativeScore> {
    @Getter @Setter
    private IAlternative alternative;

    @Getter @Setter
    private double score;

    @Getter @Setter
    private DecisionRule rule;

    public AlternativeScore(IAlternative alternative, double score, DecisionRule rule) {
        this.alternative = alternative;
        this.score = score;
        this.rule = rule;
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
        if (this == obj) return true;
        if (!(obj instanceof AlternativeScore)) return false;
        AlternativeScore other = (AlternativeScore) obj;
        return this.alternative.equals(other.alternative) && Double.compare(this.score, other.score) == 0;
    }
    

    @Override
    public int hashCode() {
        return Double.hashCode(score) * 31 + alternative.hashCode();
    }
}
