package it.univr.controlledmarkovchains.valueiteration;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import it.univr.usefulmethodsarrays.UsefulMethodsArrays;

/**
 * Main goal of this class is to provide the solution of a stochastic control problem in the setting
 * of controlled Markov chains for discrete time and discrete space, under the hypothesis that the transition
 * probabilities from one state to the other (i.e., the probabilities defining "the environment") are known.
 * Specifically, the representation of the problem with the Bellman equation  is used in order to iteratively
 * get the value function for every state, that is, the value achieved with the "best action". The best action
 * at every state is then computed as the action for which the maximum is achieved.
 * The methods providing the possible actions for every state and the best action for every state depend on the
 * specific problem, and for this reason are implemented in the derived classes. 
 *  
 * @author Andrea Mazzon
 *
 */
public abstract class ValueIteration {
	
	
	//TO BE GIVEN IN THE CONSTRUCTOR 
	
	//the possible states of the system
	private double[] states;
	
	//the final rewards for every state. They must be zero for non absorbing states.
	private double[] rewardsAtStates;
		
	//the discount factor gamma in the notes
	private double discountFactor;
	
	//it will be used to check if a state index corresponds to an absorbing state
	private List<Integer> absorbingStatesIndicesAsList;
	
	/*
	 * The iterations stop when the absolute value of the difference between the new and past values of the value
	 * function is smaller than requiredPrecision for all the entries
	 */ 
	private double requiredPrecision;
	
	
	//TO BE FILLED/COMPUTED
	
	//this array will contain the value function for every state, that is, the maximized values
	private double[] valueFunctions;
	
	//this array will contain the optimal actions for every state
	private double[] optimalActions;
		
	private int numberOfStates;
	
	//it will be udpated during the loop, and contains the past values for every state
	private double[] oldValueFunctions;
	
	//a list of double arrays containing the value functions which are computed during the loop
	private ArrayList<double[]> updatedValueFunctions = new ArrayList<double[]>();
	
	
	/**
	 * It constructs an object to solve a stochastic control problem in the setting of controlled Markov chains
	 * for discrete time and discrete space, under the hypothesis that the transition probabilities from one state
	 * to the other (i.e., the probabilities defining "the environment") are known.
	 * 
	 * @param states, the possible states of the system
	 * @param rewardsAtStates, the rewards for every state. They must be zero for non absorbing states.
	 * @param absorbingStatesIndices, the indices of states which are absorbing: for example, for the gambler problem
	 * 		  they are 0 and the last index.
	 * @param discountFactor, the discount factor gamma in the notes
	 * @param requiredPrecision, the iterations stop when the absolute value of the difference between the new and past
	 * 		  values of the value function is smaller than requiredPrecision for all the entries
	 */
	public ValueIteration(double[] states, double[] rewardsAtStates, int[] absorbingStatesIndices, double discountFactor, double requiredPrecision) {
		
		this.states = states;
		
		this.rewardsAtStates = rewardsAtStates;
		
		this.discountFactor = discountFactor; 
		
        absorbingStatesIndicesAsList = Arrays.stream(absorbingStatesIndices).boxed().toList();
						
		this.requiredPrecision = requiredPrecision; 
		
		numberOfStates = states.length;

	}
	
	/*
	 * This is a private method which is used to compute the value functions for every state and then the
	 * optimizing actions. It is the chore of the class.
	 */
	private void generateValueFunctionsAndOptimalActions() throws Exception {

		//at the beginning, the value functions are just the rewards for every state. They will then get updated
		valueFunctions = rewardsAtStates.clone();
		oldValueFunctions = rewardsAtStates.clone();

		//so we know it is bigger then requiredPrecision and the loop starts
		double differenceBetweenPastAndOldValueFunctions = Double.MAX_VALUE;
		
		while (differenceBetweenPastAndOldValueFunctions >= requiredPrecision) {

			//we update the value of the value function for every state which is not absorbing
			for (int stateIndex = 0; stateIndex < numberOfStates; stateIndex ++) {

				if (!absorbingStatesIndicesAsList.contains(stateIndex)) {

					//the possible actions for the specific state: they depend on the specific problem
					double[] actions = computeActions(states[stateIndex]);

					/*
					 * The returns for these actions, as the sum of running reward and (discounted) value function: they depend
					 * on the specific problem as well
					 */
					double[] actionReturns = computeExpectedReturnsForStateAndActions(states[stateIndex],actions);

					//the new value function for the state is the maximum between the returns
					double newValue = UsefulMethodsArrays.getMax(actionReturns);

					//we update the value function for the given state
					valueFunctions[stateIndex] = newValue;
				}
			}

			//we store it in a new record of updatedValueFunctions
			updatedValueFunctions.add(valueFunctions.clone()); 
			
			/*
			 * We check the maximum absolute difference between the new and the opld value function: if this is smaller than
			 * requiredPrecision, the loop stops
			 */
			differenceBetweenPastAndOldValueFunctions = UsefulMethodsArrays.getMaxDifference(valueFunctions, oldValueFunctions);

			//update of the old value functions.
			oldValueFunctions = valueFunctions.clone();
		}
		
		//the loop is now terminated: we get the optimal actions
		optimalActions = computeOptimalActions();

	}

	
	// This method gets called once the "final" value function is computed
	private double[] computeOptimalActions() {
		
		//one optimal action for every state
		double[] optimalActions = new double[numberOfStates];
		
		
		for (int stateIndex = 0; stateIndex < numberOfStates; stateIndex ++ ) {
			if (absorbingStatesIndicesAsList.contains(stateIndex)) { 
				//no action is possible in the absorbing states
				optimalActions[stateIndex] = (int) Double.NaN;
			} else {
			
			//the possible actions for the state
        	double[] actions = computeActions(states[stateIndex]);
        	
        	//the returns for those actions
        	double[] actionReturns = computeExpectedReturnsForStateAndActions(states[stateIndex], actions);
        	
        	//the index of the optimal action
        	int indexOfOptimalAction = UsefulMethodsArrays.getMaximizingIndex(actionReturns);
        	optimalActions[stateIndex] = actions[indexOfOptimalAction];
			}
        }
		return optimalActions;
	}
	
	
	protected double[] getOldValueFunctions() {
		//needed to calculate the return of the actions in the derived classes
		return oldValueFunctions.clone();
	}
		
	/**
	 * It returns the discount factor 
	 * 
	 * @return the discount factor
	 */
	public double getDiscountFactor() {
		return discountFactor;
	}
	
	/**
	 * It returns a double array representing the value functions for every state
	 * 
	 * @return a double array representing the value functions for every state
	 * @throws Exception 
	 */
	public double[] getValueFunctions() throws Exception {
		if (valueFunctions == null) {
			//it gets called only once!
			generateValueFunctionsAndOptimalActions();
		}
		return valueFunctions.clone();
	}
	
	/**
	 * It returns a double array representing the optimal actions providing the value functions for every state
	 * 
	 * @return a double array representing the value functions for every state
	 * @throws Exception 
	 */
	public double[] getOptimalActions() throws Exception {
		if (valueFunctions == null) {
			//it gets called only once!
			generateValueFunctionsAndOptimalActions();
		}
		return optimalActions.clone();
	}
		
	/**
	 * It returns the list of arrays of doubles recording all the updated value functions
	 * 
	 * @return the list of arrays of doubles recording all the updated value functions
	 * @throws Exception 
	 */
	@SuppressWarnings("unchecked")
	public ArrayList<double[]> getUpdatedOptimalValues() throws Exception {
		if (valueFunctions == null) {
			generateValueFunctionsAndOptimalActions();
		}
		return (ArrayList<double[]>) updatedValueFunctions.clone();
	}
	
	/**
	 * It computes and returns an array of doubles which represents the actions that are allowed for the given state
	 * @returns an array of doubles which represents the actions that are allowed for the given state
	 */
	protected abstract double[] computeActions(double state);
	
	/**
	 * It computes and returns an array of doubles which represents the (expected) returns associated to every action for a given state
	 * @returns an array of doubles which represents the actions that are allowed for the given state
	 */
	protected abstract double[] computeExpectedReturnsForStateAndActions(double state, double[] actions);
}