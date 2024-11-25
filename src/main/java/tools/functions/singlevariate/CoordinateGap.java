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
        for (int i : getCoordinates())
            for (int j = 0; j < n; j++)
                if(!getCoordinates().contains(j))
                    score += alternative.getVector()[i] - alternative.getVector()[j];

        return score;
    }

    @Override
    public double computeScore(IAlternative alternative, DecisionRule rule) {
        return computeScore(alternative);
    }
}
