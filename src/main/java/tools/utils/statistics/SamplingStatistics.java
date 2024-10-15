package tools.utils.statistics;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class SamplingStatistics {
    // The list of alternative appearance frequency in sample
    // (sorted by descending order).
    private List<Integer> alternativeFrequencies;

    // The list of corresponding alternative scores (scores of the alternative).
    private List<Double> alternativeScores;

    // The entropy of the frequencies of the alternatives in the sample. 
    private double entropy;
}
