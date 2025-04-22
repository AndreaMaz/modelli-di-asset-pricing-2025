package it.univr.controlleddiffusionprocesses;

import java.text.DecimalFormat;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import org.apache.commons.lang3.function.TriFunction;


/**
 * This class tests the implementation of the Policy Improvement Algorithm to numerically solve the Merton problem, whose analytic solution
 * is computed in Example 4.20 of the script.
 * 
 * @author Andrea Mazzon
 *
 */
public class PolicyImprovementTest {
	
	public static void main(String[] args) throws Exception {
		
		DecimalFormat formatterForControl = new DecimalFormat("0.00");
		DecimalFormat formatterForValue = new DecimalFormat("0.000");
		
		/*
		 * Parameters to define the functions for the SDE and for the rewards.
		 * They will be used also to compute the optimal control and the number for the analytic value function
		 * which is our benchmark, see the script.
		 */
		double interestRate = 0.2;
		double constantDrift = 0.3;
		double constantSigma = 0.25;
		
		double exponentForFinalRewardFunction = 0.5;		
		
		double optimalControl = (constantDrift - interestRate)/(constantSigma*constantSigma*(1-exponentForFinalRewardFunction));

		System.out.println("Analytic optimal control " +  formatterForControl.format(optimalControl));
		System.out.println();
		
		
		//functions for the SDE
		TriFunction<Double, Double, Double, Double> driftFunctionWithControl = (t,x,a) -> x*(a*(constantDrift-interestRate)+interestRate);
		TriFunction<Double, Double, Double, Double> diffusionFunctionWithControl = (t,x,a) -> x*a*constantSigma;
		
		//functions for the rewards
		TriFunction<Double, Double, Double, Double> runningRewardFunction = (t,x,a) -> 0.0;
		DoubleUnaryOperator finalRewardFunction = x -> Math.pow(x,exponentForFinalRewardFunction);
		
		//function for the left border. In our case, the left border is zero
		DoubleBinaryOperator functionLeft = (t, a) -> 0.0;
		
		//definition of the intervals
		double leftEndControlInterval = 0.0;
		double rightEndControlInterval = 6;
		double controlStep = 0.01;
		
		double leftEndSpaceInterval = 0.0;
		double rightEndSpaceInterval = 10;
		double spaceStep = 0.1;
		
		double finalTime = 3.0;
		double timeStep = 0.1;
		
		double requiredPrecision = 0.001;
		int maxNumberIterations = 20;
		
		PolicyImprovement optimizer = new PolicyImprovement(driftFunctionWithControl, diffusionFunctionWithControl, runningRewardFunction, finalRewardFunction, functionLeft,
				leftEndControlInterval,  rightEndControlInterval,  controlStep,  leftEndSpaceInterval, rightEndSpaceInterval,  spaceStep,  finalTime,  timeStep, requiredPrecision, maxNumberIterations);
		
		
		//it will be used to compute the analytic value function
		double beta = 0.5 * constantSigma * constantSigma * optimalControl * optimalControl * exponentForFinalRewardFunction * (exponentForFinalRewardFunction - 1)
				+ (constantDrift-interestRate)*exponentForFinalRewardFunction*optimalControl+interestRate*exponentForFinalRewardFunction;
		
		
		//we check the value function and the control at every combination of these time and space values
		double[] timeToCheck = {0.5, 1.0, 1.5, 2.0, 2.5};
		
		double[] spaceToCheck = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}; 
		
		
		for (double time : timeToCheck) {
			for (double space : spaceToCheck) {
				
				System.out.println("Time: " + time + " Space: " + space);
				
				double analyticValueFunction = Math.exp(time*beta)*Math.pow(space, exponentForFinalRewardFunction);
				
				System.out.println("Analytic value function " +  formatterForValue.format(analyticValueFunction));

				double approximatedValueFunction = optimizer.getValueFunctionAtTimeAndSpace(time, space);
								
				System.out.println("Approximated value function " + formatterForValue.format(approximatedValueFunction));
				
				double approximatedOptimalControl = optimizer.getOptimalControlAtTimeAndSpace(time, space);
				
				System.out.println("Approximated optimal control " + formatterForControl.format(approximatedOptimalControl));
				
				System.out.println();
			}
		}		
	}
}