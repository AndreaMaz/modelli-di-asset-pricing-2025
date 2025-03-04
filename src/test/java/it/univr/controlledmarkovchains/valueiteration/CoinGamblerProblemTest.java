package it.univr.controlledmarkovchains.valueiteration;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import net.finmath.plots.Named;
import net.finmath.plots.Plot2D;
import net.finmath.plots.Plotable2D;
import net.finmath.plots.PlotableFunction2D;


/**
 * This class tests the implementation of ValueIteration and its derived class CoinGamblerProblem.
 * 
 * @author Andrea Mazzon
 *
 */
public class CoinGamblerProblemTest {
	
	public static void main(String[] args) throws Exception {
		
		double requiredPrecision = 1e-4;

		double headProbability = 0.4;
		
		int amountOfMoneyToReach = 100;
		
		double discountFactor = 1.0;//no discount
		
		ValueIteration problemSolver = new CoinGamblerProblem(discountFactor, requiredPrecision, headProbability, amountOfMoneyToReach);

		//plot of the last value functions
		double[] valueFunctions = problemSolver.getValueFunctions();
		
		final Plot2D plotValueFunctions = new Plot2D(1, amountOfMoneyToReach-1, amountOfMoneyToReach-1, Arrays.asList(
				new Named<DoubleUnaryOperator>("Value function", x -> valueFunctions[(int) x])));
		
		plotValueFunctions.setXAxisLabel("State");
		plotValueFunctions.setYAxisLabel("Value function");
		
		plotValueFunctions.show();
		
		//plot of the last optimal actions
		double[] optimalActions = problemSolver.getOptimalActions();
			
		
		final Plot2D plotOptimalPolicy = new Plot2D(1, amountOfMoneyToReach-1, amountOfMoneyToReach-1, Arrays.asList(
				new Named<DoubleUnaryOperator>("Optimal action", x -> optimalActions[(int) x])));
		
		plotOptimalPolicy.setXAxisLabel("State");
		plotOptimalPolicy.setYAxisLabel("Optimal money investment on head");
		
		plotOptimalPolicy.show();
		
		//dynamic plot of the updated value functions
		
		//we start from the very first ones..
		ArrayList<double[]> updatedValues = problemSolver.getUpdatedOptimalValues();
		
		double[] firstValueFunctions = updatedValues.get(0);//note the get(int index) method of ArrayList
		
		final Plot2D plotUpdatedValueFunctions = new Plot2D(1, amountOfMoneyToReach-1, amountOfMoneyToReach-1, Arrays.asList(
				new Named<DoubleUnaryOperator>("Updated value function", x -> firstValueFunctions[(int) x])));

		plotUpdatedValueFunctions.setXAxisLabel("State");
		plotUpdatedValueFunctions.setYAxisLabel("Updated value function, iteration " + 1);
		plotUpdatedValueFunctions.show();
		Thread.sleep(4000);

		//..and then we dynamically print the next ones, until the final ones
		for (int iterationIndex = 1; iterationIndex < updatedValues.size(); iterationIndex ++) {
			
			//this is what we want to plot, at every iteration of the while loop
			double[] updatedValueFunction = updatedValues.get(iterationIndex);
			
			final List<Plotable2D> plotables = Arrays.asList(
					new PlotableFunction2D(1, amountOfMoneyToReach-1, amountOfMoneyToReach-1,
							new Named<DoubleUnaryOperator>("Updated value function", x -> updatedValueFunction[(int) x]),null));

			plotUpdatedValueFunctions.update(plotables);//in this way, all the plots are in the same figure
			plotUpdatedValueFunctions.setYAxisLabel("Updated value function, iteration " + (iterationIndex+1));
			Thread.sleep(4000); 
		}
	}
}
