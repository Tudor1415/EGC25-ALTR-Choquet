package sampling;

import java.util.List;

import tools.rules.DecisionRule;

public interface Sampler {
    List<DecisionRule> sample();
}
