package it.univr.neuralnetworks;


import java.text.DecimalFormat;

import net.finmath.exception.CalculationException;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloBlackScholesModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * This class tests the implementation of the class DeepHedging to get the final portfolio values for an hedging strategy (so, profits and losses only
 * coming from the strategies minus payoff to pay plus price at which the derivative is sold) when the underlying is given by Black-Scholes.
 * It trains the network with 10000 trajectories and test it with 10.
 * 
 * @author Andrea Mazzon
 *
 */
public class DeepHedgingTest {


	public static void main(String[] args) throws CalculationException {
		//we use it to print the final values
		DecimalFormat formatter = new DecimalFormat("0.00");

		//model (i.e., underlying) parameters
		double initialValue = 1;
		double riskFreeRate = 0.0;
		double volatility = 0.3;

		//option parameters
		double maturity = 1.0;	
		double strike = initialValue;

		//the price at which the option is sold (in this case we know it directly)
		double blackScholesPrice = AnalyticFormulas.blackScholesOptionValue(initialValue, riskFreeRate, volatility, maturity, strike);

		//Monte Carlo time discretization parameters
		double initialTime = 0.0;
		double timeStep = 0.01;
		int numberOfTimeSteps = (int) (maturity/timeStep);

		TimeDiscretization times = new TimeDiscretizationFromArray(initialTime, numberOfTimeSteps, timeStep);

		//simulation parameters
		int numberOfPaths = 10000;
		int seed = 1897;//this will be used also for the network
		
		BrownianMotion ourDriver = new BrownianMotionFromMersenneRandomNumbers(times, 1 /* numberOfFactors */, numberOfPaths, seed);


		//we construct an object of type MonteCarloBlackScholesModel: we use it to generate the trajectories of the underlying price
		AssetModelMonteCarloSimulationModel pricesGenerator = new MonteCarloBlackScholesModel(initialValue, riskFreeRate, volatility, ourDriver);		


		//network parameters
		int numberOfNodesForFirstLayer = 50;
		int numberOfNodesForSecondLayer = 50;
		int numberOfEpochs = 30;
		double learningRate = 0.01;
		
		DeepHedging tester = new DeepHedging(pricesGenerator, numberOfNodesForFirstLayer,numberOfNodesForSecondLayer,numberOfEpochs, learningRate,
				strike, blackScholesPrice, seed);

		//parameters for the price generator of the test set
		int numberOfPathsToTest = 10;		
		int newSeed = 2000;

		double[] portfolioValuesTest = tester.getPortfolioValuesForTesting(numberOfPaths, newSeed);

		
		
		System.out.println("Portfolio values for the test:");

		for (int simulationIndexForTest = 0; simulationIndexForTest<numberOfPathsToTest; simulationIndexForTest++) {
			System.out.println(formatter.format(portfolioValuesTest[simulationIndexForTest]));
		}
	}
}