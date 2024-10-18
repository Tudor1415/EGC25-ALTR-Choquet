package sampling;

import java.util.List;

import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.normalization.Normalizer;
import tools.rules.DecisionRule;

public interface ISampler extends Sampler {

    int getMaximumIterations();

    void setMaximumIterations(int maximumIterations);

    int getTopK();

    void setTopK(int topK);

    Dataset getDataset();

    void setDataset(Dataset dataset);

    CertaintyFunction getOutRankingCertainty();

    void setOutRankingCertainty(CertaintyFunction outRankingCertainty);

    ISinglevariateFunction getScoringFunction();

    void setScoringFunction(ISinglevariateFunction scoringFunction);

    String[] getMeasureNames();

    void setMeasureNames(String[] measureNames);

    Normalizer getNormalizer();

    List<DecisionRule> sample();

    double getValidRuleScore(DecisionRule rule);

}