package experiments.configs;

import java.io.IOException;

import com.google.gson.JsonSyntaxException;

import tools.ranking.RankingsProvider;

public interface QuerySelectionConfig {

    public void loadFromFile(String filePath) throws IOException, JsonSyntaxException;

    public void setUp() throws Exception;

    public RankingsProvider getRankingsProvider();

    public String getName();
}