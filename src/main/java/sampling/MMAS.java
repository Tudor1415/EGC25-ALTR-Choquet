package sampling;

import java.util.List;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.rules.DecisionRule;
import tools.normalization.Normalizer;
import tools.functions.multivariate.CertaintyFunction;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.MultivariateToSinglevariate;

public class MMAS {
    // The square root of the maximum iterations
    private @Getter @Setter int maximumIterations;
    private @Getter @Setter int topK = 1;
    private @Getter @Setter Dataset dataset;
    private @Getter @Setter CertaintyFunction certaintyFunction;
    private @Getter MultivariateToSinglevariate scoringFunction;
    private @Getter SMAS singleVariateSampler;
    private @Setter @Getter String[] measureNames;
    private boolean isUncertainty = false;

    public MMAS(int maximumIterations, int topK, Dataset dataset, CertaintyFunction certaintyFunction,
            String[] measureNames) {
        this.maximumIterations = maximumIterations;
        this.topK = topK;
        this.dataset = dataset;
        this.certaintyFunction = certaintyFunction;
        this.scoringFunction = new MultivariateToSinglevariate(certaintyFunction.getName() + "Singlevariate",
                certaintyFunction, dataset.getRandomValidRules(10, 1e-6d, measureNames), 1, isUncertainty);

        this.measureNames = measureNames;
        BatchSampler sampler = new BatchSampler(10, dataset, getScoringFunction(), measureNames, 1);
        this.singleVariateSampler = sampler;
    }

    public List<DecisionRule[]> sample() {
        for (int i = 0; i < maximumIterations; i++) {
            DecisionRule topRule = getSingleVariateSampler().sample().get(0);
            getScoringFunction().addToHistory(topRule.getAlternative(), topRule);
        }

        return getScoringFunction().getTopK(topK);
    }

    public void setScoringFunction(ISinglevariateFunction approxFunction) {
        getCertaintyFunction().setScoreFunction(approxFunction);
        this.scoringFunction = new MultivariateToSinglevariate(certaintyFunction.getName() + "Singlevariate",
                certaintyFunction, dataset.getRandomValidRules(2, 1e-6d, measureNames), 10, isUncertainty);
        getSingleVariateSampler().setScoringFunction(this.scoringFunction);
    }

    public Normalizer getNormalizer() {
        return getSingleVariateSampler().getNormalizer();
    }

    public void setNormalizationTechnique(NormalizationMethod norm) {
        getSingleVariateSampler().setNormalizationTechnique(norm);
    }

    public boolean isUncertainty() {
        return this.scoringFunction.getIsUncertainty();
    }

    public void setUncertainty(boolean isUncertainty) {
        this.isUncertainty = isUncertainty;
        this.scoringFunction.setIsUncertainty(isUncertainty);
    }

}
