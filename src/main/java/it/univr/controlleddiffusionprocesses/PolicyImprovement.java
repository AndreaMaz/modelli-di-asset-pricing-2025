package it.univr.controlleddiffusionprocesses;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.lang3.function.TriFunction;

import it.univr.pdesolvers.CrankNicholsonPDESolver;
import it.univr.usefulmethodsarrays.UsefulMethodsArrays;

/**
 * Main goal of this class is to provide an approximated solution to the class of optimal stochastic
 * control problems studied in Section 4 of the notes, for space dimension equal to one.
 * We do that via the Policy Improvement Algorithm, see Section 4.7.
 * We use a PDE solver to numerically approximate the solution of the PDEs for given controls. In particular,
 * we use the class CrankNicholsonPDESolver.
 * 
 * @author Andrea Mazzon
 *
 */
public class PolicyImprovement {
	
	//functions of time, space and control. You see four doubles because they map three doubles in one double
	private TriFunction<Double, Double, Double, Double> driftFunctionWithControl;
	private TriFunction<Double, Double, Double, Double> diffusionFunctionWithControl;
	private TriFunction<Double, Double, Double, Double> runningRewardFunction;

	//this is function of space only
	private DoubleUnaryOperator finalRewardFunction;

	//it will be given to the PDE solver
	private DoubleBinaryOperator conditionAtLeftBoundary;
	
	//the interval A where we look for the possible controls. A priori it is given by all real numbers, but we have to bound it here
	private double leftEndControlInterval;
	private double rightEndControlInterval;
	private double controlStep;

	//the interval where the state process can take values. Again, for our algorithm it has to be bounded, of course.
	private double leftEndSpaceInterval;
	private double rightEndSpaceInterval;
	private double spaceStep;
	private int numberOfSpaceSteps;

	private double finalTime;
	private double timeStep;
	private int numberOfTimeSteps;

	/*
	 * The iterations stop if the norm of the difference between the new and past values of the matrix of the value function is smaller
	 * than requiredPrecision
	 */ 
	private double requiredPrecision;

	//they also stop if we reach the maximum number of iterations
	private int maxNumberIterations;
	
	//it will contain the values of the optimal control, updated at every iteration. Time is on the rows, space on the columns
	private double[][] updatedOptimalControl;

	//it will contain the value function, updated at every iteration. Time is on the rows, space on the columns
	private double[][] updatedValueFunction;


	/**
	 * It constructs on object to solve a general stochastic optimal control problem in continuous time, with one-dimensional domain for the controlled process.
	 * It employs the Policy Improvement Algorithm.
	 * 
	 * @param driftFunctionWithControl, the function b(t,x,a) of time, space and control
	 * @param diffusionFunctionWithControl, the function sigma(t,x,a) of time, space and control
	 * @param runningRewardFunction, the function f(t,x,a) of time, space and control
	 * @param finalRewardFunction, the function g(x) 
	 * @param conditionAtLeftBoundary, the condition at the right boundary of the space domain we want to consider. The condition at
	 * 		  the right boundary is not needed.
	 * @param leftEndControlInterval, the left boundary of the interval where controls are taken
	 * @param rightEndControlInterval, the right boundary of the interval where controls are taken
	 * @param controlStep, the discretization step of the control interval
	 * @param leftEndSpaceInterval, the left boundary of the space interval 
	 * @param rightEndSpaceInterval, the right boundary of the space interval 
	 * @param spaceStep, the discretization step of the space interval
	 * @param finalTime, the time at which the final reward is potentially given
	 * @param timeStep, the discretization step of the time interval [0, finalTime]
	 * @param requiredPrecision, the iterations stop if the norm of the difference between the new and past values of the matrix of the value function is smaller
	 * 		  than this
	 * @param maxNumberIterations, the iterations also stop if we reach the maximum number of iterations
	 */
	public PolicyImprovement(TriFunction<Double, Double, Double, Double> driftFunctionWithControl, TriFunction<Double, Double, Double, Double> diffusionFunctionWithControl,
			TriFunction<Double, Double, Double, Double> runningRewardFunction, DoubleUnaryOperator finalRewardFunction, DoubleBinaryOperator conditionAtLeftBoundary,
			double leftEndControlInterval, double rightEndControlInterval, double controlStep, double leftEndSpaceInterval,
			double rightEndSpaceInterval, double spaceStep, double finalTime, double timeStep, double requiredPrecision, int maxNumberIterations) {
		
		this.driftFunctionWithControl = driftFunctionWithControl;
		this.diffusionFunctionWithControl = diffusionFunctionWithControl;
		this.runningRewardFunction = runningRewardFunction;
		this.finalRewardFunction = finalRewardFunction;

		this.conditionAtLeftBoundary = conditionAtLeftBoundary;
		
		this.leftEndControlInterval = leftEndControlInterval;
		this.rightEndControlInterval = rightEndControlInterval;
		this.controlStep = controlStep;

		this.leftEndSpaceInterval = leftEndSpaceInterval;
		this.rightEndSpaceInterval = rightEndSpaceInterval;
		this.spaceStep = spaceStep;

		this.finalTime = finalTime;
		this.timeStep = timeStep;

		numberOfTimeSteps = (int) (finalTime/timeStep);
		numberOfSpaceSteps = (int) ((rightEndSpaceInterval-leftEndSpaceInterval)/spaceStep);	
		
		this.requiredPrecision = requiredPrecision;
		this.maxNumberIterations = maxNumberIterations;
	}


	/*
	 * This is the core of the class: here we apply indeed the Policy Improvement Algorithm by starting from a given matrix of controls which will be
	 * iteratively updated based on the solution of the PDE for the past optimal controls. 
	 */
	private void computeSolutionAndOptimalControl() throws Exception {

		updatedOptimalControl = new double[numberOfTimeSteps][numberOfSpaceSteps + 1];
		
		//the first matrix of optimal controls has same values for all rows and columns
		for (int rowIndex = 0; rowIndex < numberOfTimeSteps; rowIndex ++) {
			//these values are the middle point of the control interval
			Arrays.fill(updatedOptimalControl[rowIndex], (rightEndControlInterval-leftEndControlInterval)/2);
		}
		
		//we construct the object to solve the first PDE, for the first control
		CrankNicholsonPDESolver solver = new CrankNicholsonPDESolver(spaceStep,  timeStep,  leftEndSpaceInterval,  rightEndSpaceInterval,  finalTime,
				driftFunctionWithControl, diffusionFunctionWithControl, runningRewardFunction, finalRewardFunction, conditionAtLeftBoundary, 
				updatedOptimalControl);

		//we update the value function..
		updatedValueFunction = solver.getSolution().clone();

		//..and based on the updated value function we also update the optimal control 
		updatedOptimalControl = getMaximizingControl(updatedValueFunction).clone();

		//just to be sure that we enter the while loop
		double differenceNorm = Double.MAX_VALUE;

		int iterationCounter = 0;//the while loop stops if it reaches maxNumberIterations
		while (differenceNorm > requiredPrecision & iterationCounter < maxNumberIterations) {
			
			//it is used to compute the norm of the difference between the matrices of the new and the past solution
			double[][] oldSolution = updatedValueFunction.clone();

			//object to solve the new PDE. The only thing that changes, at every iteration, is updatedOptimalControl
			solver = new CrankNicholsonPDESolver(spaceStep,  timeStep,  leftEndSpaceInterval,  rightEndSpaceInterval,  finalTime,
					driftFunctionWithControl, diffusionFunctionWithControl, runningRewardFunction, finalRewardFunction, conditionAtLeftBoundary, 
					updatedOptimalControl);

			//we update the value function..
			updatedValueFunction = solver.getSolution().clone();
			
			//..and based on the updated value function we also update the optimal control 
			updatedOptimalControl = getMaximizingControl(updatedValueFunction).clone();
			
			//the norm of a matrix is here computed as the maximum sum of the elements of its rows divided by the number of columns
			differenceNorm = UsefulMethodsArrays.getNormDifference(updatedValueFunction, oldSolution);
			iterationCounter ++;
		}
	}

	/*
	 * At the k-th iteration of the Policy Improvement Algorithm we implement, this method returns a matrix whose element of row i and column
	 * is the control a^k(t[i],x[j]) that maximizes 1/2(sigma(t[i],x[j],a))^2 * \partial_xx v^k + (b(t[i],x[j],a)) * \partial_x v^k + f()t[i],x[j],a),
	 * where v^k is the value function computed at the k-th iteration. The derivatives are approximated via final differences. 
	 */
	private double[][] getMaximizingControl(double[][] currentValueFunction){

		//matrix to fill
		double[][] maximizingControls = new double[numberOfTimeSteps][numberOfSpaceSteps + 1];

		int numberOfControls = (int) ((rightEndControlInterval - leftEndControlInterval)/controlStep) + 1;

		//all the possible controls. For every time and space, we choose the maximizing one.
		double[] controls = IntStream.range(0, numberOfControls).mapToDouble(i -> leftEndControlInterval + i * controlStep).toArray();
		double time = timeStep;
		
		for (int timeIndex = 1; timeIndex <= numberOfTimeSteps; timeIndex ++) {
			
			double space = leftEndSpaceInterval;
			
			for (int spaceIndex = 0; spaceIndex <= numberOfSpaceSteps; spaceIndex ++) {
				double firstSpaceDerivative = computeFirstDerivative(timeIndex, spaceIndex);
				double secondSpaceDerivative = computeSecondDerivative(timeIndex, spaceIndex);
				double[] valuesForControls = new double[numberOfControls];
				
				//we compute the values of all the controls..
				for (int controlIndex = 0; controlIndex < numberOfControls; controlIndex ++) {
					double drift = driftFunctionWithControl.apply(time, space, controls[controlIndex]);
					double volatility = diffusionFunctionWithControl.apply(time, space, controls[controlIndex]);
					valuesForControls[controlIndex] = 0.5 * volatility * volatility * secondSpaceDerivative + drift * firstSpaceDerivative
							+ runningRewardFunction.apply(time, space, controls[controlIndex]);
				}
				
				//..and take the control that maximizes them
				maximizingControls[timeIndex - 1][spaceIndex]=controls[UsefulMethodsArrays.getMaximizingIndex(valuesForControls)];
				space += spaceStep;
			}
						
			time += timeStep;
		}
		return maximizingControls;
	}

	/*
	 * This method computes the approximated first derivative of updatedValueFunction at the point in time and space
	 * determined by timeIndex and spaceIndex
	 */
	private double computeFirstDerivative(int timeIndex, int spaceIndex) {

		if (spaceIndex == 0) {
			return (updatedValueFunction[timeIndex][1]-updatedValueFunction[timeIndex][0])/spaceStep;
		} else if (spaceIndex == numberOfSpaceSteps) {
			return (updatedValueFunction[timeIndex][numberOfSpaceSteps]-updatedValueFunction[timeIndex][numberOfSpaceSteps - 1])/spaceStep;
		}
		else {
			return (updatedValueFunction[timeIndex][spaceIndex + 1]-updatedValueFunction[timeIndex ][spaceIndex - 1])/(2*spaceStep);
		}
	}
	
	/*
	 * This method computes the approximated second derivative of updatedValueFunction at the point in time and space
	 * determined by timeIndex and spaceIndex
	 */
	private double computeSecondDerivative(int timeIndex, int spaceIndex) {

		if (spaceIndex == 0) {
			return (updatedValueFunction[timeIndex][2]-2*updatedValueFunction[timeIndex][1]+updatedValueFunction[timeIndex][0])/(spaceStep*spaceStep);

		} else if (spaceIndex == numberOfSpaceSteps) {
			return (updatedValueFunction[timeIndex][numberOfSpaceSteps]-2*updatedValueFunction[timeIndex][numberOfSpaceSteps - 1]
					+updatedValueFunction[timeIndex][numberOfSpaceSteps - 2])/(spaceStep*spaceStep);
		}
		else {
			return (updatedValueFunction[timeIndex][spaceIndex + 1]-2*updatedValueFunction[timeIndex][spaceIndex]
					+updatedValueFunction[timeIndex][spaceIndex - 1])/(spaceStep*spaceStep);
		}
	}
	
	/**
	 * It returns the value function as a matrix of doubles.
	 * 
	 * @return It returns the value function as a matrix of doubles. The time remaining to final time is on rows, space on columns
	 * @throws Exception
	 */
	public double[][] getValueFunction() throws Exception {
		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedValueFunction.clone();
	}

	/**
	 * It returns the optimal controls as a matrix of doubles.
	 * 
	 * @return It returns the optimal controls as a matrix of doubles. The time remaining to final time is on rows, space on columns
	 * @throws Exception
	 */
	public double[][] getOptimalControl() throws Exception {
		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedOptimalControl.clone();
	}


	/**
	 * It returns the value function at a given time as a vector of doubles. The time is here meant as 
	 * time remaining to final time
	 * @param time, the time when the value function has to be computed. It his here meant as time remaining to final time
	 * @return the vector of the value function for that time
	 * @throws Exception
	 */
	public double[] getValueFunctionAtGivenTime(double time) throws Exception {

		int timeIndex = (int) Math.round(time / timeStep);

		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedValueFunction[timeIndex];
	}

	/**
	 * It returns the optimal control at a given time as a vector of doubles. The time is here meant as 
	 * time remaining to final time
	 * @param time, the time when the value function has to be computed. It his here meant as time remaining to final time
	 * @return the vector of the value function for that time
	 * @throws Exception
	 */
	public double[] getOptimalControlAtGivenTime(double time) throws Exception {

		int timeIndex = (int) Math.round(time / timeStep);

		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedOptimalControl[timeIndex - 1];
	}

	/**
	 * It returns the value function at a given time and a given space value. The time is here meant as 
	 * time remaining to final time
	 * @param time, the time when the value function has to be computed. It his here meant as time remaining to final time
	 * @param space, the value of x
	 * @return the value function for that time and space
	 * @throws Exception
	 */
	public double getValueFunctionAtTimeAndSpace(double time, double space) throws Exception {

		int timeIndex = (int) Math.round(time / timeStep);
		int spaceIndex = (int) Math.round((space - leftEndSpaceInterval) / spaceStep);
		
		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedValueFunction[timeIndex][spaceIndex];
	}

	/**
	 * It returns the optimal control at a given time and a given space value. The time is here meant as 
	 * time remaining to final time
	 * @param time, the time when the value function has to be computed. It his here meant as time remaining to final time
	 * @param space, the value of x
	 * @return the optimal control for that time and space
	 * @throws Exception
	 */
	public double getOptimalControlAtTimeAndSpace(double time, double space) throws Exception {

		int timeIndex = (int) Math.round(time / timeStep);
		int spaceIndex = (int) Math.round((space - leftEndSpaceInterval) / spaceStep);

		if (updatedValueFunction == null) {
			computeSolutionAndOptimalControl();
		}
		return updatedOptimalControl[timeIndex - 1][spaceIndex];
	}
}