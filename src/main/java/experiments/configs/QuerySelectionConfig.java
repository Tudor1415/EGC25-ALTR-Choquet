package experiments.configs;

import tools.ranking.RankingsProvider;

public interface QuerySelectionConfig {

    public void setUp() throws Exception;

    public RankingsProvider getRankingsProvider();
}