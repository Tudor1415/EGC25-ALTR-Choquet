DATASETS = bank credit toms connect mushroom
exp_active = results/active_learning
exp_norm = results/active_normalization
exp_sample = results/sampling_experiment

dat-files: $(patsubst %,data/dat-files/%.dat,$(DATASETS))

data/dat-files/%.dat: data/datasets/%.csv
	python data/format.py $< data/dat-files/$*.dat data/mapping-files/$*.map

folds:
	python scripts/prepare_data.py data/dat-files/ 10 data/folds

python-install:
	@which python3 > /dev/null || (echo "Python is not installed. Please install Python 3 first." && exit 1)
	pip install -r requirements.txt

r-install:
	@which R > /dev/null || (echo "R is not installed. Please install R first." && exit 1)
	Rscript -e "if (!require('kappalab')) install.packages('kappalab', repos='https://cloud.r-project.org')"
	Rscript -e "if (!require('jsonlite')) install.packages('jsonlite', repos='https://cloud.r-project.org')"

exp_active:
	rm -rf ${exp_active}
	mkdir -p ${exp_active}/samples
	mvn exec:java -Dexec.mainClass="experiments.ExperimentActiveLearning"

exp_norm:
	rm -rf ${exp_norm}
	mkdir -p ${exp_norm}/samples
	mvn exec:java -Dexec.mainClass="experiments.ExperimentActiveNormalization"

exp_sample:
	rm -rf ${exp_sample}
	mkdir -p ${exp_sample}/samples
	mvn exec:java -Dexec.mainClass="experiments.SamplingMethodExperiment"

plot_exp_sampling:
	python scripts/plot_sampling.py 

plot_exp_active:
	python scripts/plot_active_learning.py results/active_learning/samples --cumulative

plot_exp_norm:
	python scripts/plot_active_learning.py results/active_normalization/samples