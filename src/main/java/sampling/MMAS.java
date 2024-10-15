package sampling;

import java.util.List;

import lombok.Getter;
import lombok.Setter;
import tools.data.Dataset;
import tools.functions.multivariate.CertaintyFunction;
import tools.functions.singlevariate.ISinglevariateFunction;
import tools.functions.singlevariate.MultivariateToSinglevariate;
import tools.normalization.Normalizer;
import tools.normalization.Normalizer.NormalizationMethod;
import tools.rules.DecisionRule;

public class MMAS {
    // The square root of the maximum iterations
    private @Getter @Setter int maximumIterations;
    private @Getter @Setter int topK = 1;
    private @Getter @Setter Dataset dataset;
    private @Getter @Setter CertaintyFunction certaintyFunction;
    private @Getter MultivariateToSinglevariate scoringFunction;
    private @Getter SMAS singleVariateSampler;
    private @Setter @Getter String[] measureNames;

    public MMAS(int maximumIterations, int topK, Dataset dataset, CertaintyFunction certaintyFunction,
            String[] measureNames) {
        this.maximumIterations = maximumIterations;
        this.topK = topK;
        this.dataset = dataset;
        this.certaintyFunction = certaintyFunction;
        this.scoringFunction = new MultivariateToSinglevariate(certaintyFunction.getName() + "Singlevariate",
                certaintyFunction, dataset.getRandomValidRules(10, 1e-6d, measureNames), 1);

        this.measureNames = measureNames;
        SMAS sampler = new SMAS(10, dataset, getScoringFunction(), measureNames, 1);
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
                certaintyFunction, dataset.getRandomValidRules(2, 1e-6d, measureNames), 100);
    }

    public Normalizer getNormalizer() {
        return getSingleVariateSampler().getNormalizer();
    }

    public void setNormalizationTechnique(NormalizationMethod norm) {
        getSingleVariateSampler().setNormalizationTechnique(norm);
    }
}
