package tools.utils.statistics;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class DatasetStatistics {
    private String name;
    private int foldIdx;
    private int nbTransactions;
    private double smoothCounts;
    private String[] measureNames;
}
