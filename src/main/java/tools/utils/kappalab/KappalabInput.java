package tools.utils.kappalab;

import lombok.Getter;
import lombok.Setter;
import tools.alternatives.IAlternative;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.gson.annotations.Expose;

/**
 * Represents the input parameters for the Kappalab Choquet integral.
 *
 * @param alternatives       Vectors of the alternatives.
 * @param preferences
 *                           <p>
 *                           Represents preferences where each element is an
 *                           array indicating
 *                           a preference relation.
 *                           <p/>
 *                           <p>
 *                           For example, [2, 1, 0.5] means that the
 *                           alternative of index 2 is preferred to the
 *                           alternative of index 1
 *                           with a delta value of 0.5.
 *                           <p/>
 * @param k                  The k-additivity of the model.
 * @param approachType       Represents the approach type used, such as "mv"
 *                           (Minimum Variance)
 *                           or "gls" (Generalized Least Squares).
 * @param significantFigures Precision (number of significant figures).
 */
@Getter
@Setter
public class KappalabInput {

    /** Vectors of the alternatives */
    @Expose(serialize = true, deserialize = true)
    private List<double[]> alternatives;

    /**
     * Hash map providing each alternative with an index by order of addition to the
     * map.
     */
    @Expose(serialize = false, deserialize = false)
    private Map<IAlternative, Integer> alternativeMap = new HashMap<>();

    @Expose(serialize = false, deserialize = false)
    private Map<Integer, Set<Double>> scoreMap = new HashMap<>();

    /**
     * Represents preferences where each element is an array indicating a preference
     * relation.
     */
    @Expose(serialize = true, deserialize = true)
    private List<Number[]> preferences;

    /** The k-additivity of the Choquet integral */
    @Expose(serialize = true, deserialize = true)
    private int kAdditivity;

    /**
     * Represents the approach type used, such as "Minimum Variance"
     * or "Generalized Least Squares".
     */
    @Expose(serialize = true, deserialize = true)
    private String approachType;

    /** Precision (number of significant figures) */
    @Expose(serialize = true, deserialize = true)
    private int significantFigures;

    /**
     * Constructs a new instance of {@code KappalabInput} with the specified
     * parameters.
     *
     * @param alternatives Vectors of the alternatives.
     * @param preferences  Preference information.
     * @param kAdditivity  The k-additivity of the Choquet integral.
     * @param approachType The approach type used, e.g., "Minimum Variance" or
     *                     "Generalized Least Squares".
     */
    public KappalabInput(List<IAlternative> alternatives, List<Number[]> preferences, int kAdditivity,
            String approachType) {
        this.alternatives = alternatives.stream()
                .map(alternative -> (double[]) alternative.getVector())
                .collect(Collectors.toList());
        this.preferences = preferences;
        this.kAdditivity = kAdditivity;
        this.approachType = approachType;
        significantFigures = 3;
    }

    /**
     * Constructs a new instance of {@code KappalabInput} with minimal
     * parameters.
     * 
     */
    public KappalabInput(int kAdditivity, String approachType) {
        this.alternatives = new ArrayList<>();
        this.preferences = new ArrayList<Number[]>();
        this.kAdditivity = kAdditivity;
        this.approachType = approachType;
        significantFigures = 3;
    }

    /**
     * Add alternative if not in alternative Map and return its index.
     * 
     * @param a alternative
     * @return idx of the alternative
     */
    private int addAlternative(IAlternative a) {
        for(IAlternative alternative : alternativeMap.keySet())
            if (alternative.equals(a)) {
                return alternativeMap.get(a);
            }
        alternativeMap.put(a, alternativeMap.size());
        alternatives.add(a.getVector());
        return alternativeMap.size() - 1;
    }

    /**
     * Adds a preference relation between two alternatives along with the specified
     * delta value.
     *
     * @param a     The first alternative in the preference relation.
     * @param b     The second alternative in the preference relation.
     * @param delta The delta value indicating the preference strength>
     */
    public void addPreference(IAlternative a, IAlternative b, double delta, Double[] scores) {
        int aIndex = addAlternative(a) + 1;
        int bIndex = addAlternative(b) + 1;
        preferences.add(new Number[] { aIndex, bIndex, delta });

        Set<Double> aScores = scoreMap.computeIfAbsent(aIndex, k -> new HashSet<>());
        aScores.add(scores[0]);
        if (aScores.size() >= 2) {
            System.out.println("Warning: Multiple scores detected for alternative aIndex: " + (aIndex + 1));
        }

        Set<Double> bScores = scoreMap.computeIfAbsent(bIndex, k -> new HashSet<>());
        bScores.add(scores[1]);
        if (bScores.size() >= 2) {
            System.out.println("Warning: Multiple scores detected for alternative bIndex: " + (bIndex + 1));
        }
    }
}
