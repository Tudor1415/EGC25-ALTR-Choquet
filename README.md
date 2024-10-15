# Active Learning to Rank based-Gibbs Sampling for Classification Rule Discovery

## Experimental requirements

Before running the experiments, you need to follow these instructions:

- Install R and the following R packages: `kappalab` and `jsonlite` using `make r-install`
- Install Python 3 with `requirements.txt` using `make python-install`
- Create JAR with dependencies (Maven required): `mvn package`
- Extract the datasets.zip archive in the data/datasets folder
- Run `make dat-files`
- Run `make folds`

## Running the experiments

To run the experiments, run `make exp_active OR exp_norm OR exp_sample`

## Plotting

To get the plots run `make plot_--INSERT_EXP_NAME--`.
