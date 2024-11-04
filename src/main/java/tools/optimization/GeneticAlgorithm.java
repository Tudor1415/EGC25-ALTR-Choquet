package tools.optimization;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import tools.utils.kappalab.MobiusCapacity;

public class GeneticAlgorithm {
    private int populationSize;
    private int numGenerations;
    private double crossoverRate;
    private double mutationRate;
    private int nbCriteria;
    private int kAdditivity;
    private ArrayList<MobiusCapacity> population;
    private Random random;

    public GeneticAlgorithm(int populationSize, int numGenerations, double crossoverRate, double mutationRate, int nbCriteria, int kAdditivity) {
        this.populationSize = populationSize;
        this.numGenerations = numGenerations;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.nbCriteria = nbCriteria;
        this.kAdditivity = kAdditivity;
        this.population = new ArrayList<>();
        this.random = new Random();
        initializePopulation();
    }

    // Initialize the population with random MobiusCapacity instances
    private void initializePopulation() {
        for (int i = 0; i < populationSize; i++) {
            double[] randomCapacities = new double[(int) Math.pow(2, nbCriteria)];
            for (int j = 0; j < randomCapacities.length; j++) {
                randomCapacities[j] = random.nextDouble();
            }
            MobiusCapacity capacity = new MobiusCapacity(nbCriteria, kAdditivity, randomCapacities);
            population.add(capacity);
        }
    }

    // Evaluate the population using ChoquetMobiusScoreFunction
    private void evaluatePopulation(IAlternative alternative) {
        for (MobiusCapacity capacity : population) {
            ChoquetMobiusScoreFunction scoreFunction = new ChoquetMobiusScoreFunction(capacity);
            double score = scoreFunction.computeScore(alternative);
            capacity.setLoss(score); // Assuming MobiusCapacity has a setLoss method
        }
    }

    // Perform selection using tournament selection
    private MobiusCapacity selectParent() {
        int tournamentSize = 3;
        ArrayList<MobiusCapacity> tournament = new ArrayList<>();
        for (int i = 0; i < tournamentSize; i++) {
            tournament.add(population.get(random.nextInt(populationSize)));
        }
        return Collections.min(tournament, (m1, m2) -> Double.compare(m1.getLoss(), m2.getLoss()));
    }

    // Perform crossover between two parent MobiusCapacity instances
    private MobiusCapacity crossover(MobiusCapacity parent1, MobiusCapacity parent2) {
        double[] offspringCapacities = new double[parent1.getOrderedCapacityValues().length];
        for (int i = 0; i < offspringCapacities.length; i++) {
            offspringCapacities[i] = random.nextDouble() < crossoverRate ? parent1.getOrderedCapacityValues()[i] : parent2.getOrderedCapacityValues()[i];
        }
        return new MobiusCapacity(nbCriteria, kAdditivity, offspringCapacities);
    }

    // Perform mutation on a MobiusCapacity
    private void mutate(MobiusCapacity capacity) {
        if (random.nextDouble() < mutationRate) {
            double[] capacities = capacity.getOrderedCapacityValues();
            int index = random.nextInt(capacities.length);
            capacities[index] += (random.nextDouble() * 2 - 1) * 0.1; // Mutation range is 0.1
            capacities[index] = Math.max(0, Math.min(1, capacities[index])); // Keep within [0, 1]
            capacity = new MobiusCapacity(nbCriteria, kAdditivity, capacities); // Recreate MobiusCapacity
        }
    }

    // Run the genetic algorithm
    public MobiusCapacity run(IAlternative alternative) {
        evaluatePopulation(alternative);

        for (int generation = 0; generation < numGenerations; generation++) {
            ArrayList<MobiusCapacity> newPopulation = new ArrayList<>();

            // Generate new population
            for (int i = 0; i < populationSize; i++) {
                MobiusCapacity parent1 = selectParent();
                MobiusCapacity parent2 = selectParent();
                MobiusCapacity offspring = crossover(parent1, parent2);
                mutate(offspring);
                newPopulation.add(offspring);
            }

            population = newPopulation;
            evaluatePopulation(alternative);
        }

        // Return the best MobiusCapacity found
        return Collections.min(population, (m1, m2) -> Double.compare(m1.getLoss(), m2.getLoss()));
    }
}
