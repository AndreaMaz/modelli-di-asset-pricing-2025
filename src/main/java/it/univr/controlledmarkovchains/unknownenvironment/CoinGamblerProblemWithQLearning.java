package it.univr.controlledmarkovchains.unknownenvironment;

import java.util.stream.IntStream;

import org.jblas.util.Random;

/**
 * The main contribution of this class is to provide the solution of the gambler problem when the probability of getting head
 * is not known. It does it by extending the class QLearning, providing the implementation of its abstract methods.
 * 
 * @author Andrea Mazzon
 *
 */
public class CoinGamblerProblemWithQLearning extends QLearning {

	
	private double headProbability;
	private int amountOfMoneyToReach;

	/**
	 * It constructs an object to compute the solution of the gambler problem with unknown head probability.
	 * 
	 * @param discountFactor: the discount factor gamma in the notes
	 * @param headProbability, the probability to get head
	 * @param amountOfMoneyToReach, the amount that the capital process must hit in order for the gambler to win the bet
	 * @param numberOfEpisodes, the number of loops from an initial state until an absorbing state 
	 * @param learningRate, the learning rate lambda that enters in the update rule 
	 * 		  Q(x,a) <- Q(x,a) + lambda * (f^a(x)+gamma*max_{b in A(y)} Q(y,b)-Q(x,a))
	 * @param explorationProbability, the probability that an action for a given state is randomly chosen
	 */
	public CoinGamblerProblemWithQLearning(double headProbability, double discountFactor, int amountOfMoneyToReach, int numberOfEpisodes, double learningRate, double explorationProbability) {
		super(IntStream.concat(IntStream.generate(() -> 0).limit(amountOfMoneyToReach), IntStream.of(1) ).asDoubleStream().toArray(), 
				new int[] {0, amountOfMoneyToReach}, //the indices of the absorbing states
				discountFactor,
				new double[amountOfMoneyToReach+1][amountOfMoneyToReach-1],//the array of running rewards: it is just made of zeros
				numberOfEpisodes, learningRate, explorationProbability);
		this.headProbability = headProbability;
		this.amountOfMoneyToReach = amountOfMoneyToReach;
	}

	
	@Override
	protected int getNumberOfActions() {
		return amountOfMoneyToReach - 1;//the possible actions which makes sense to consider are 1, 2, ..., amountOfMoneyToReach - 1
	}

	@Override
	protected int[] computePossibleActionsIndices(int stateIndex) {
		/*
		 * Possible actions are (1,2,..,n) where n is the minimum between the capital (we cannot go negative) and the capital
		 * needed to reach amountOfMoneyToReach (it does not make sense to invest more). The index is the action minus 1: the index of action "1" is 0,
		 * the index of action "2" is 1, and so on.
		 */        
        return IntStream.range(0, Math.min(stateIndex, amountOfMoneyToReach - stateIndex)).toArray();
	}

	@Override
	protected int generateStateIndex(int oldStateIndex, int actionIndex) {
		double randomResult = Random.nextDouble();
		int action = actionIndex+1;//the action is the index plus 1!
		if (randomResult < headProbability) {
			return (oldStateIndex + action);
		}
		return (oldStateIndex - action);
	}


}
