package it.univr.controlledmarkovchains.unknownenvironment;


import java.util.Arrays;
import java.util.List;
import java.util.Random;

import it.univr.usefulmethodsarrays.UsefulMethodsArrays;


/**
 * Main goal of this class is to provide the solution of a stochastic control problem in the setting
 * of controlled Markov chains for discrete time and discrete space, under the hypothesis that the transition
 * probabilities from one state to the other (i.e., the probabilities defining "the environment") are not known.
 * Specifically, the representation of the problem with the Bellman equation is used in order to iteratively
 * get the Q-value for every state and every action, by the update rule
 * Q(x,a) <- Q(x,a) + lambda * (f^a(x)+gamma*max_{b in A(y)} Q(y,b)-Q(x,a)),
 * where f^a(x) is the running reward from choosing action a at state x, y is the next state where one lands from
 * x with action a,  A(y) is the set of possible actions in y, lambda is the learning rate and gamma is the discount
 * factor.
 * The procedure is repeated over multiple "episodes": every episode ends when one of the absorbing states is reached.
 *  
 * @author Andrea Mazzon
 *
 */
public abstract class QLearning {

	//the final rewards for every state. They must be zero for non absorbing states.
	private double[] rewardsAtStates;

	//the discount factor gamma in the notes
	private double discountFactor;

	//the running rewards, as a matrix: runningRewards[i][j] is the running reward for the i-th state and for the j-th action
	private double[][] runningRewards;

	//this array will contain the value function for every state, that is, the maximized values
	private double[] valueFunctions;

	//the indices of the optimal actions for every state
	private int[] optimalActionsIndices;

	//the Q-values, as a matrix: currentQValue[i][j] is the Q-value for the i-th state and for the j-th action
	private double[][] currentQValue;

	private int numberOfStates;

	private int numberOfEpisodes;

	/*
	 * The learning rate lambda that enters in the update rule 
	 * Q(x,a) <- Q(x,a) + lambda * (f^a(x)+gamma*max_{b in A(y)} Q(y,b)-Q(x,a))
	 */
	private double learningRate;

	/*
	 * A parameter in [0,1] that characterizes the exploration probability: an action at a given state x is
	 * randomly chosen in the set of possible actions for x with probability equal to explorationProbability, and is instead
	 * chosen as the maximizing action for the Q-value in x with probability equal to 1 - explorationProbability
	 */
	private double explorationProbability;

	//used to generate the random numbers to determine exploration or exploitation and to choose the random action for exploration
	private Random generator = new Random();


	//it will be used to check if a state index corresponds to an absorbing state
	private List<Integer> absorbingStatesIndicesAsList;

	/**
	 * It constructs an object to solve a stochastic control problem in the setting of controlled Markov chains for discrete time and discrete space,
	 * under the hypothesis that the transition probabilities from one state to the other (i.e., the probabilities defining "the environment") are not known.
	 * 
	 * @param rewardsAtStates, the final rewards for every state. They must be zero for non absorbing states.
	 * @param discountFactor, the discount factor gamma in the notes
	 * @param runningRewards, the running rewards, as a matrix: runningRewards[i][j] is the running reward for the i-th state and for the j-th action
	 * @param numberOfEpisodes, the number of loops from an initial state until an absorbing state 
	 * @param learningRate, the learning rate lambda that enters in the update rule 
	 *            Q(x,a) <- Q(x,a) + lambda * (f^a(x)+gamma*max_{b in A(y)} Q(y,b)-Q(x,a))
	 * @param explorationProbability, the probability that an action for a given state is randomly chosen
	 */
	public QLearning(double[] rewardsAtStates, int[] absorbingStatesIndices, double discountFactor, double[][] runningRewards, int numberOfEpisodes,
			double learningRate, double explorationProbability) {
		this.rewardsAtStates = rewardsAtStates;
		numberOfStates = rewardsAtStates.length;
		absorbingStatesIndicesAsList = Arrays.stream(absorbingStatesIndices).boxed().toList();
		this.discountFactor = discountFactor;
		this.runningRewards = runningRewards;
		this.numberOfEpisodes = numberOfEpisodes; 
		this.learningRate = learningRate;
		this.explorationProbability = explorationProbability;
	}

	/*
	 * 
	 * This is a private method which is used to fill the currentQValue matrix, which represents the Q-value for every state and action.
	 * The value function and the index of the best action for the i-th state are then computed as max_{j}currentQValue[i,j] and
	 * argmax_{j}currentQValue[i,j], respectively. This method is the chore of the class.
	 */
	private void generateOptimalValueAndPolicy() {

		valueFunctions = new double[numberOfStates];
		optimalActionsIndices = new int[numberOfStates];

		int numberOfActions = getNumberOfActions();
		currentQValue = new double[numberOfStates][numberOfActions];

		/*
		 * We give the initial Q-values: for actions which are not permitted for a given state, we set the Q-value to minus Infinity.
		 * For actions which are permitted, we make the initial value of the Q-value equal to the final reward for the associated state,
		 * independently of the action
		 */
		for (int stateIndex = 0; stateIndex < numberOfStates; stateIndex ++) {
			//the index of the actions which are allowed for that state
			int[] possibleActionsIndices = computePossibleActionsIndices(stateIndex);

			//we make it a list because then it's easier to check if it contains the given action indices
			List<Integer> possibleActionsIndicesAsList = Arrays.stream(possibleActionsIndices).boxed().toList();

			//the column index is the action index
			for (int actionIndex = 0; actionIndex < numberOfActions; actionIndex ++) {
				currentQValue[stateIndex][actionIndex]=possibleActionsIndicesAsList.contains(actionIndex) ? rewardsAtStates[stateIndex] : Double.NEGATIVE_INFINITY;
			}
		}

		//now we go through the episodes

		//any episode starts from a randomly chosen state and terminates when hitting an absorbing state
		for (int episodeIndex = 0; episodeIndex < numberOfEpisodes; episodeIndex ++) {

			//we generate a possible state
			int temptativeStateIndex = generator.nextInt(numberOfStates);

			//if it is an absorbing state, we want to generate another one, and so on
			while (absorbingStatesIndicesAsList.contains(temptativeStateIndex)) {
				temptativeStateIndex = generator.nextInt(numberOfStates);
			}

			//finally, we get the state which is not absorbing
			int stateIndex = temptativeStateIndex;

			//this will be updated
			int chosenActionIndex;

			while (true) {//it ends when we land in an absorbing state

				if (generator.nextDouble()< explorationProbability){//exploration: randomly chosen action
					int[] possibleActionsIndices = computePossibleActionsIndices(stateIndex);
					chosenActionIndex = possibleActionsIndices[generator.nextInt(possibleActionsIndices.length)];
				} else {//exploitation: one maximizing action                           
					chosenActionIndex = UsefulMethodsArrays.getMaximizingIndex(currentQValue[stateIndex]);
				}

				/*
				 * The index of the new state, randomly picked in a way which depends on the action and on the state.
				 * Since the way is chosen depends on the specific problem, the method is abstract and gets implemented
				 * in the derived classes.
				 */
				int newStateIndex = generateStateIndex(stateIndex, chosenActionIndex);

				if (absorbingStatesIndicesAsList.contains(newStateIndex)) {
					//if we land at an absorbing state, there is no possible action to be taken: the value is equal to the reward
					currentQValue[stateIndex][chosenActionIndex] = currentQValue[stateIndex][chosenActionIndex] +
							learningRate * (discountFactor*rewardsAtStates[newStateIndex]-currentQValue[stateIndex][chosenActionIndex]) ;
					break; //we exit the while loop
				}

				//if we are not landed in an absorbing state, we now want to compute the maximum Q-value for the new state 
				double maximumForGivenStateIndex = UsefulMethodsArrays.getMax(currentQValue[newStateIndex]);

				//update
				currentQValue[stateIndex][chosenActionIndex] = currentQValue[stateIndex][chosenActionIndex] +
						learningRate * (runningRewards[stateIndex][chosenActionIndex] + discountFactor*maximumForGivenStateIndex-currentQValue[stateIndex][chosenActionIndex]) ;


				stateIndex = newStateIndex;
			}
		}

		//now we have run all the episodes, so we have our "final" currentQValue matrix. We then compute the value functions and the optimal actions

		for (int stateIndexAtTheEnd = 0; stateIndexAtTheEnd < numberOfStates; stateIndexAtTheEnd ++) {
			if (absorbingStatesIndicesAsList.contains(stateIndexAtTheEnd)) { 
				//no action is possible in the absorbing states
				valueFunctions[stateIndexAtTheEnd] = rewardsAtStates[stateIndexAtTheEnd];
				optimalActionsIndices[stateIndexAtTheEnd] = (int) Double.NaN;
			} else {
				valueFunctions[stateIndexAtTheEnd] = UsefulMethodsArrays.getMax(currentQValue[stateIndexAtTheEnd]);
				optimalActionsIndices[stateIndexAtTheEnd] = UsefulMethodsArrays.getMaximizingIndex(currentQValue[stateIndexAtTheEnd]);
			}
		}
	}


	protected double[][] getCurrentQValue() {
		return currentQValue.clone();
	}

	/**
	 * It returns a double array representing the value functions for every state
	 * 
	 * @return a double array representing the value functions for every state
	 */
	public double[] getValueFunctions() {
		if (currentQValue == null) {
			generateOptimalValueAndPolicy();
		}
		return valueFunctions.clone();
	}

	/**
	 * It returns a double array representing the optimal actions providing the value functions for every state
	 * 
	 * @return a double array representing the value functions for every state
	 */
	public int[] getOptimalActionsIndices() {
		if (valueFunctions == null) {
			generateOptimalValueAndPolicy();
		}
		return optimalActionsIndices.clone();
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
	 * It returns the total number of possible actions (indipendent from the state)
	 * @return the total number of possible actions (indipendent from the state)
	 */
	protected abstract int getNumberOfActions();

	/**
	 * It computes and returns an array of integers which represents the indices of the actions that are allowed for the given state
	 * @return an array of integers which represents the indices of the actions that are allowed for the given state
	 */
	protected abstract int[] computePossibleActionsIndices(int stateIndex);

	/**
	 * It (randomly) generates the index of the next state, based on the old state index and on the chosen action index
	 * 
	 * @param oldStateIndex
	 * @param actionIndex
	 * @return the index of the next state
	 */
	protected abstract int generateStateIndex(int oldStateIndex, int actionIndex); 
}