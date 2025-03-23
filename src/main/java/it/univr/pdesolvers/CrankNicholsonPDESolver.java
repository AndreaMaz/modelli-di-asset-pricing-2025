package it.univr.pdesolvers;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.lang3.function.TriFunction;

import it.univr.usefulmethodsarrays.UsefulMethodsArrays;

/**
 * Main goal of this class is to numerically approximate the solution of the PDE
 *
 * \partial_t v(t,x) = 1/2 (\volatilityFunction(t,x,a(t,x)))^2 \partial_xx v(t,x)+b(t,x,a(t,x)) \partial_x v(t,x)+f(t,x,a(t,x)), (t,x) \in (0,T] x R,
 * v(0,x)=g(x), x \in R.
 * 
 * Note that here the time variable t can be interpreted as the difference between final time T and current time in the PDE coming from
 * the control problem.
 * 
 * In particular, the method of Crank-Nicholson is used to solve the PDE.
 * 
 * @author Andrea Mazzon
 *
 */
public class CrankNicholsonPDESolver {

	//time steps for space and time
	private double dx;
	private double dt;
	
	//left boundary of the interval for the space. It will be needed in getSolutionForGivenTimeAndSpace
	private double xMin;

	//vectors for space and time
	private double[] x;
	private double[] t;


	private int numberOfSpaceSteps;
	private int numberOfTimeSteps;
	
	//this can be interpreted as the final condition in the control problem (remember that time is flipped)
	private DoubleUnaryOperator initialCondition;
	
	//condition at the left boundary
	private DoubleBinaryOperator conditionAtLeftBoundary;

	//they will be used to compute the solution from one time step to the other
	private int currentTimeIndex;
	private double currentTime;

	//they will store the solution at past and current time when computing the solution going forward
	private double[] solutionAtPastTime;
	private double[] solutionAtCurrentTime;
	
	//it will store the whole approximated solution. Time is on the rows, space on the columns
	private double[][] solution;

	//functions of time, space and control. You see four doubles because they map three doubles in one double
	private TriFunction<Double, Double, Double, Double> driftFunction;
	private TriFunction<Double, Double, Double, Double> volatilityFunction;
	private TriFunction<Double, Double, Double, Double> functionForKnownTerm;//it is the running reward function for the control problem

	//the matrix of the controls. We have controlMatrix[i][j]=a(t_i,x_j). We need it when computing the three functions above
	private double[][] controlMatrix;

	//the y will me multiplied to the approximated first and second derivatives, respectively
	private double multiplyTermFirstDerivative = 0.5 * dt / dx;
	private double multiplyTermSecondDerivative = dt / (dx * dx);


	/**
	 * It constructs an object to compute the approximated solution of the PDE
	 *  \partial_t v(t,x) = 1/2 (\volatilityFunction(t,x,a(t,x)))^2 \partial_xx v(t,x)+b(t,x,a(t,x)) \partial_x v(t,x)
	 *   +f(t,x,a(t,x)), (t,x) \in (0,T] x R,
	 * v(0,x)=g(x), x \in R,
	 * via Crank-Nicholson.
	 * 
	 * @param dx, the space step
	 * @param dt, the time step
	 * @param xMin, the left boundary of the space domain we want to consider
	 * @param xMax, the right boundary of the space domain we want to consider
	 * @param tMax, the final time
	 * @param driftFunction, the function b(t,x,a(t,x)) of time, space and control (which can depend on time and space)
	 * @param volatilityFunction, the function sigma(t,x,a(t,x)) of time, space and control (which can depend on time and space)
	 * @param functionForKnownTerm, the function f(t,x,a(t,x)) of time, space and control (which can depend on time and space)
	 * @param initialCondition, this is the final reward function for the optimal control problem (time here is flipped)
	 * @param conditionAtLeftBoundary, the condition at the right boundary of the space domain we want to consider. The condition at
	 * the right boundary is not needed.
	 * @param controlMatrix, the matrix of the controls. We have controlMatrix[i][j]=a(t_i,x_j)
	 */
	public CrankNicholsonPDESolver(double dx, double dt, double xMin, double xMax, double tMax,
			TriFunction<Double, Double, Double, Double> driftFunction,
			TriFunction<Double, Double, Double, Double> volatilityFunction,
			TriFunction<Double, Double, Double, Double> functionForKnownTerm,
			DoubleUnaryOperator initialCondition, DoubleBinaryOperator conditionAtLeftBoundary,
			 double[][] controlMatrix) {

		this.dx = dx;
		this.dt = dt;
		this.xMin = xMin;
		this.initialCondition = initialCondition;
		this.conditionAtLeftBoundary = conditionAtLeftBoundary;

		numberOfSpaceSteps = (int) Math.ceil((xMax - xMin) / dx);
		numberOfTimeSteps = (int) Math.ceil(tMax / dt);
		
		// Create equi-spaced space discretization
		x = IntStream.range(0, numberOfSpaceSteps + 1).mapToDouble(i -> xMin + i * dx).toArray();
		t = IntStream.range(0, numberOfTimeSteps + 1).mapToDouble(i -> i * dt).toArray();

		this.driftFunction = driftFunction;
		this.volatilityFunction = volatilityFunction;
		this.functionForKnownTerm = functionForKnownTerm;
		this.controlMatrix = controlMatrix;
		
		//WE have dt because we multiply everything by dt when computing the numerical scheme
		multiplyTermFirstDerivative = 0.5 * dt / dx;
		multiplyTermSecondDerivative = dt / (dx * dx);

		currentTimeIndex = 1;
	}


	//this is the chore of the class: it computes the approximated solution going forward from one time to the other
	private void solveAndSave() {
		computeMatrixForSystem();//the matrix that appears in the linear system
		
		solution = new double[numberOfTimeSteps + 1][numberOfSpaceSteps + 1];

		//the solution at initial time: given by the initial condition. It is needed to get the solution at current time
		solutionAtPastTime = IntStream.range(0, numberOfSpaceSteps + 1).mapToDouble(i -> initialCondition.applyAsDouble(x[i])).toArray();
		
		solution[0] = solutionAtPastTime.clone();//it is always safer to use clones!

		//a for loop that goes forward in time
		for (int i = 1; i <= numberOfTimeSteps; i++) {
			currentTime = t[currentTimeIndex];//it is needed in the method getSolutionAtCurrentTime()
			solutionAtCurrentTime = getSolutionAtCurrentTime().clone();
			solutionAtPastTime = solutionAtCurrentTime.clone();
			solution[i] = solutionAtCurrentTime.clone();
			currentTimeIndex ++;
		}
	}

	/*
	 * It computes the matrix of the system. It will be called at any call of the getSolutionAtCurrentTime() because
	 * the coefficients can depend on time.
	 */
	private double[][] computeMatrixForSystem() {

		//it will be defined as a tri-diagonal matrix. 
		double[][] matrixForTheSystem = new double[numberOfSpaceSteps - 1][numberOfSpaceSteps - 1];
		double[] diagonal = new double[numberOfSpaceSteps - 1];
		double[] lowerDiagonal = new double[numberOfSpaceSteps - 2];
		double[] upperDiagonal = new double[numberOfSpaceSteps - 2];

		//first, we have to define the diagonal, the lower diagonal and the upper diagonal
		for (int spaceIndex = 0; spaceIndex < numberOfSpaceSteps - 2; spaceIndex++) {

			double currentSpaceVariable = x[spaceIndex + 1];		
			double currentDrift = driftFunction.apply(currentTime, currentSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex] );
			double currentSigma = volatilityFunction.apply(currentTime, currentSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex]);

			double nextSpaceVariable = x[spaceIndex + 2];
			double nextDrift = driftFunction.apply(currentTime, nextSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex + 1]);
			double nextSigma = volatilityFunction.apply(currentTime, nextSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex + 1]);

			
			diagonal[spaceIndex] = 1 + 0.5 * multiplyTermSecondDerivative * currentSigma * currentSigma;

			lowerDiagonal[spaceIndex] = -0.5 * multiplyTermSecondDerivative * nextSigma * nextSigma
					+ multiplyTermFirstDerivative * nextDrift;

			upperDiagonal[spaceIndex] = -0.5 * multiplyTermSecondDerivative * currentSigma * currentSigma
					- multiplyTermFirstDerivative * currentDrift;
		}

		//the last element of the diagonal (it has one element more)
		diagonal[numberOfSpaceSteps - 2] = 1 + 0.5 * multiplyTermSecondDerivative
				* volatilityFunction.apply(currentTime, x[numberOfSpaceSteps - 1], controlMatrix[currentTimeIndex - 1][numberOfSpaceSteps - 1])
		* volatilityFunction.apply(currentTime, x[numberOfSpaceSteps - 1], controlMatrix[currentTimeIndex - 1][numberOfSpaceSteps - 1]);

		matrixForTheSystem = new double[numberOfSpaceSteps - 1][numberOfSpaceSteps - 1];
		
		//now, having the three diagonals, we contruct the matrix
		for (int i = 0; i < numberOfSpaceSteps - 2; i++) {
			matrixForTheSystem[i][i] = diagonal[i];
			matrixForTheSystem[i][i + 1] = 0.5 * upperDiagonal[i];
			matrixForTheSystem[i + 1][i] = 0.5 * lowerDiagonal[i];
		}
		
		matrixForTheSystem[numberOfSpaceSteps - 2][numberOfSpaceSteps - 2] = diagonal[numberOfSpaceSteps - 2];
		return matrixForTheSystem;
	}


	/*
	 * It computes the known term of the system. It will be called at any call of the getSolutionAtCurrentTime() because
	 * the coefficients can depend on time.
	 */
	private double[] computeKnownTermForSystem() {

		double[] firstDerivatives = new double[numberOfSpaceSteps - 1];
		double[] secondDerivatives = new double[numberOfSpaceSteps - 1];

		//the first derivatives of the solution at past time, multiplied by dt
		for (int i = 1; i < numberOfSpaceSteps; i++) {
			//central differences
			firstDerivatives[i-1] = multiplyTermFirstDerivative * (solutionAtPastTime[i + 1] - solutionAtPastTime[i - 1]);
			secondDerivatives[i-1] = multiplyTermSecondDerivative * (solutionAtPastTime[i + 1] - 2 * solutionAtPastTime[i] + solutionAtPastTime[i - 1]);
		}


		double[] knownTerm = new double[numberOfSpaceSteps - 1];

		//basically it works as before for the diagonals
		for (int spaceIndex = 0; spaceIndex < numberOfSpaceSteps - 1; spaceIndex++) {

			double currentSpaceVariable = x[spaceIndex + 1];
			double currentDrift = driftFunction.apply(currentTime, currentSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex] );
			double currentSigma = volatilityFunction.apply(currentTime, currentSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex]);
			double currentFunctionForKnownTerm =  functionForKnownTerm.apply(currentTime, currentSpaceVariable, controlMatrix[currentTimeIndex - 1][spaceIndex]);

			double addingTerm = 0.5 * secondDerivatives[spaceIndex] * currentSigma * currentSigma + firstDerivatives[spaceIndex] * currentDrift;
			knownTerm[spaceIndex] = solutionAtPastTime[spaceIndex + 1] + 0.5 * addingTerm + currentFunctionForKnownTerm;
		}

		//the first and last elements have one factor more
		knownTerm[0] += 0.5 * solutionAtPastTime[0]
				* (0.5 * multiplyTermSecondDerivative * volatilityFunction.apply(currentTime, x[1], controlMatrix[currentTimeIndex - 1][1])* volatilityFunction.apply(currentTime, x[1], controlMatrix[currentTimeIndex - 1][1])
						- multiplyTermFirstDerivative * driftFunction.apply(currentTime, x[1], controlMatrix[currentTimeIndex - 1][1]));

		knownTerm[numberOfSpaceSteps - 2] += 0.5 * solutionAtPastTime[numberOfSpaceSteps]
				* (0.5 * multiplyTermSecondDerivative * volatilityFunction.apply(currentTime, x[numberOfSpaceSteps - 1], controlMatrix[currentTimeIndex - 1][numberOfSpaceSteps - 1])* volatilityFunction.apply(currentTime, x[1], controlMatrix[currentTimeIndex - 1][numberOfSpaceSteps - 1])
						+ multiplyTermFirstDerivative * driftFunction.apply(currentTime, x[numberOfSpaceSteps - 1], controlMatrix[currentTimeIndex - 1][numberOfSpaceSteps - 1]));

		return knownTerm;
	}

	//this is another important method: it computes the solution at current time based the solution at past time
	private double[] getSolutionAtCurrentTime() {

		double currentTime = t[currentTimeIndex];

		double[] solutionAtCurrentTime = new double[solutionAtPastTime.length];

		double[][] matrixForTheSystem = computeMatrixForSystem();
		double[] knownTerm = computeKnownTermForSystem();

		//it determine all the elements of the solution, except for the ones at the borders
		double[] solutionToTheLinearSystem = UsefulMethodsArrays.solveLinearSystem(matrixForTheSystem, knownTerm);

		for (int i = 1; i < numberOfSpaceSteps; i++) {
			solutionAtCurrentTime[i] = solutionToTheLinearSystem[i-1];
		} 

		solutionAtCurrentTime[0] = conditionAtLeftBoundary.applyAsDouble(x[0], currentTime);

		/*
		 * The problem here is that we don't have a clear idea of the approximated form of the solution at the right boundary
		 * (i.e., for large space variable x). However, since we suppose the coefficients of the PDE to be Lipschitz and then with growth
		 * no more than linear in x, we can guess the same for the solution. That is, the limit of the second derivative for large x should be zero.
		 * We impose this. 
		 */
		solutionAtCurrentTime[numberOfSpaceSteps] = 2*solutionAtCurrentTime[numberOfSpaceSteps-1] - solutionAtCurrentTime[numberOfSpaceSteps-2];

		return solutionAtCurrentTime;
	}


	/**
	 * It returns the approximated solution for given time and space
	 * @param time
	 * @param space
	 * @return the solution at specified time and space
	 */
	public double getSolutionForGivenTimeAndSpace(double time, double space) {

		if (solution == null) {
			solveAndSave();
		}

		int timeIndex = (int) Math.round(time / dt);
		int spaceIndex = (int) Math.round((space - xMin) / dx);

		return solution[timeIndex][spaceIndex];
	}

	/**
	 * It returns the solution of the PDE as a matrix. 
	 * @return the solution of the PDE as a matrix. Time is on the rows, space on the columns
	 */
	public double[][] getSolution() {
		if (solution == null) {
			solveAndSave();
		}
		return solution;
	}



}
