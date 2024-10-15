package tools.functions.singlevariate.OWA;

import org.junit.Test;

import tools.alternatives.Alternative;

public class OWALexminTest {

    @Test
    public void computeScore() {
        String[] measureNames = { "yuleQ", "cosine", "kruskal", "pavillon", "certainty" };

        OWALexmin scoreFunction = new OWALexmin(0.01, measureNames.length);

        double[][] alternativeArrays = {
                { 0.9999968357731404, 0.0076825153645359336, 0.0, 0.65748284684768, 0.9999967577674125 },
                { 0.9999992095954628, 0.015365030729064668, 0.0, 0.6574833413175118, 0.9999991906595234 },
                { 0.9999984183197085, 0.010864717421662328, 0.0, 0.65748317649407, 0.9999983796946762 },
                { 0.9999998425423647, 0.019321182485584135, 0.0016556291390728477, 0.9999996719688696,
                        0.999999488761781 },
                { 0.9999996843092837, 0.013662139156104413, 8.278145695364238e-4, 0.9999993423224797,
                        0.999998975899405 },
                { 0.7032360111586672, 0.02794925520603022, 0.0, 0.5799193873930187, 0.6183687666579225 },
                { 0.9999994733544136, 0.018818242584195795, 0.0, 0.6574833962586959, 0.9999994609813194 }
        };

        for (int i = 0; i < alternativeArrays.length - 1; i++) {
            double[] alternativeArray = alternativeArrays[i];
            Alternative alternative = new Alternative(alternativeArray);

            // Assuming scoreFunction is an existing object with a computeScore method.
            double score = scoreFunction.computeScore(alternative);
            System.out.println("Score for alternative " + i + ": " + score);
        }

    }
}
