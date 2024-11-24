package tools.functions.singlevariate;

import java.util.Set;

import lombok.Getter;
import lombok.Setter;
import tools.rules.DecisionRule;
import tools.alternatives.IAlternative;

@Getter
@Setter
public class CoordinateGap implements ISinglevariateFunction {

    private String Name = "CoordinateGap";
    private Set<Integer> coordinates;

    /**
     * This function maximizes the gap between one coordinate and the others;
     * 
     * @param coordinate The coordinate that stands out.
     */
    public CoordinateGap(Set<Integer> coordinates) {
        this.coordinates = coordinates;
    }

    @Override
    public double computeScore(DecisionRule rule) {
        return computeScore(rule.getAlternative());
    }

    @Override
    public double computeScore(IAlternative alternative) {
        double score = 0.0;

        int n = alternative.getVector().length;
        for (int i = 0; i < n; i++) {
            if (getCoordinates().contains(i))
                score += alternative.getVector()[i];
            else
                score -= alternative.getVector()[i];
        }

        return score;
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(alternative);
    }
}
