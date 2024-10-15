package tools.utils.results;

import java.util.List;
import java.util.Map;

import lombok.Data;

/** 
 * Logs the results of the active learning experiment. 
 * It is used to be written in a file in Json format. 
 */
@Data
public class ResultExpActive {
    private double timeToLearn;
    private boolean timeOut;
    private Map<String, List<Double>> metricValues;
    private Map<String, List<List<Double>>> distributions;
    private int nbIterations;
    private String oracle;
    private String learningAlgorithm;
    private String dataset;
    private int foldIdx;
    private String[] errorMessages;
    private double errorProbability;
}



