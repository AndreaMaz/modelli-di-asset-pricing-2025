package it.univr.neuralnetworks;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.ShiftVertex;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex.Op;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import net.finmath.exception.CalculationException;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.time.TimeDiscretization;


/**
 * The main task of this class is to find a strategy to optimally hedge a call option written on a one-dimensional
 * underlying, via neural networks. More on this problem can be found in Section 5 of the script: here we take the case
 * with no transaction costs, one-dimensional underlying and call option. In particular, at any time the optimal strategy is
 * computed by a neural network with two hidden layers with specified number of nodes. We construct a neural network which is the
 * juxtaposition of these networks and train it with a number of simulated trajectories of the underlying price.
 * The choice of the loss function leads to minimize the variance of the final portfolio values, see Example 5.5 of the script.
 * The class has then a public method that makes the user specify a number of trajectories simulated with a different seed
 * with respect to the one used for the training set. The final portfolio values for those trajectories based on
 * the strategies learnt by the network and applied to the new trajectories are then returned.
 * 
 * @author Andrea Mazzon
 *
 */
public class DeepHedging {

	/*
	 * ComputationGraph is a class of deeplearning4j which allows to construct a neural network in a more flexible way than
	 * MultiLayerConfiguration, which we have seen in the bubble example. Here we need it because our network structure is a
	 * bit more complex: for example, the loss function depends not only on the final choice but on all the strategies in time.
	 * Basically, this neural network will be made of (numberOfTimes-1) sub-neural networks, each of which is responsible to
	 * compute the strategy at the given time. These networks have two hidden layers with number of nodes specified by the user
	 */
	private ComputationGraph network;

	//this will generate the trajectories for training the network
	private AssetModelMonteCarloSimulationModel pricesGenerator;

	//these will be got from pricesGenerator
	private int numberOfTimes;
	private TimeDiscretization times;
	private int numberOfSimulations;

	//paremeters for the network: they will be given in the constructor
	private int seedForNetwork;
	private int numberOfNodesForFirstLayer;
	private int numberOfNodesForSecondLayer;
	private int numberOfEpochs;
	private double learningRate;

	//option parameters: they will be given in the constructor
	private double strike;
	private double optionPrice;


	/**
	 * It constructs an object to get an optimal hedging strategy via neural networks
	 * 
	 * @param pricesGenerator: AssetModelMonteCarloSimulationModel object to generate trajectories to train the network.
	 * A clone with modified number of trajectories and seed is used to get the test values
	 * @param numberOfNodesForFirstLayer: the number of nodes given to the first layer of the sub-networks, every time 
	 * @param numberOfNodesForSecondLayer: the number of nodes given to the second layer of the sub-networks, every time 
	 * @param numberOfEpochs: the number of epochs to train the big network
	 * @param learningRate: the learning rate to train the big network
	 * @param strike: the strike of the option
	 * @param optionPrice: the price for which the option is sold
	 * @param seedForNetwork: the seed to train the network (we use the Adam algorithm, which involves some stochasticity)
	 */
	public DeepHedging(AssetModelMonteCarloSimulationModel pricesGenerator, int numberOfNodesForFirstLayer,
			int numberOfNodesForSecondLayer, int numberOfEpochs, double learningRate, double strike, double optionPrice, int seedForNetwork) {

		this.pricesGenerator = pricesGenerator;

		this.numberOfNodesForFirstLayer = numberOfNodesForFirstLayer;
		this.numberOfNodesForSecondLayer = numberOfNodesForSecondLayer;
		this.numberOfEpochs = numberOfEpochs;

		this.strike = strike;
		this.optionPrice = optionPrice;	

		//the time discretization and the number of simulations are directly got from the AssetModelMonteCarloSimulationModel object
		times = pricesGenerator.getTimeDiscretization();
		numberOfTimes = times.getNumberOfTimeSteps();

		numberOfSimulations = pricesGenerator.getNumberOfPaths();

	}



	// In this method, we construct the big network as the "union" of sub-neural networks
	private void constructNetwork() {

		/*
		 * Here we specify that our network will be trained using the Adam algorithm as updater to find the optimal parameters,
		 * with specified learning rate and seed. Here it is enough to know that GraphBuilder is a class that allows
		 * to specify how to build an object of type ComputationGraph. 
		 */
		GraphBuilder builder = new NeuralNetConfiguration.Builder().updater(new Adam(learningRate)).seed(seedForNetwork)
				.graphBuilder();

		/*
		 * Based on this, from now on we have to physically construct the network. We will do that by creating:
		 * - layers: they allow to go forward and, practically in this case, to find the strategies once their parameters
		 * are trained
		 * - vertices, not to be confused with nodes: vertices are used to give new inputs (for example, the new prices at
		 * every time) and to make computations out of these inputs (for example, the difference between two prices and its 
		 * multiplication by the amount bought or sold).
		 * For both layers and vertices, we need strings in order to identify them: once we have strings, we can apply operations
		 * to these objects. 
		 * Here we give the strings for the first layers and vertices, at time 0. 
		 */

		//it will be "price0" (we could directly write price0 but this is more consistent with what we do after)
		String stringForPrice = "price" + 0; 

		//same thing for these ones

		String stringForFirstHiddenLayer = "firstHiddenLayer" + 0;
		String stringForSecondHiddenLayer = "secondHiddenLayer" + 0;

		//"strategy" will always mean how much we buy or sell
		String stringForStrategy = "strategy" + 0;

		/*
		 * "strategyValue" will always mean how much we gain/lose from the single strategy decided at time t_n, when we see
		 * the new price at time t_{n+1}
		 */
		String stringForStrategyValue;


		//The cumulative strategy value at given time will be equal to the sum of all strategyValues up to that time..
		String stringForOldCumulativeStrategyValue = "cumulativeStrategyValue" + 0;

		//..so: the new one will be the old one plus strategyValue
		String stringForCumulativeStrategyValue;

		String stringForIncrement;

		/*
		 * Here below we construct the first inputs and layers of the big network: this specification here basically
		 * constitutes the first sub-network
		 */

		/*
		 * addInputs(String inputName) is a method which basically tells Java that later, when we will effectively
		 * train and test the network, we will give an input identified by the string specified.
		 * We will then have as many different inputs as the number of times we call the method addInputs in the
		 * network construction. For any different input, as many values will be given as the size of the training set.
		 * The order with which we give the different inputs must be the same as the one with which we call the corresponding
		 * methods. See what we do in the construction of inputsArrayTrain in the trainNetwork() method.
		 * 
		 */

		//the first input is quite dummy: it's just the starting value of profits and losses coming from the strategy, i.e., zero. 
		builder.addInputs(stringForOldCumulativeStrategyValue)
		//the second one is the initial price of the underlying
		.addInputs(stringForPrice)
		/*
		 * Once we have the price, we construct the first and second layer of the first sub-network. So this sub-network 
		 * has one input node, which is the price. Note the syntax: the first argument is the string which identifies the layer,
		 * the second one the structure of the layer (how many inputs, how many outputs, which activation function) and the third
		 * one the string of the layer or -as in this case- of the input on which the layer depends.
		 */
		.addLayer(stringForFirstHiddenLayer, new DenseLayer.Builder().nIn(1).nOut(numberOfNodesForFirstLayer)
				.activation(new ActivationReLU()).build(), stringForPrice)
		//here the number of inputs must be equal to the number of outputs of the layer above, otherwise we get an exception
		.addLayer(stringForSecondHiddenLayer, new DenseLayer.Builder().nIn(numberOfNodesForFirstLayer).nOut(numberOfNodesForSecondLayer)
				.activation(new ActivationReLU()).build(), stringForFirstHiddenLayer)
		/*
		 * This layer outputs the first strategy, i.e., the first amount of money we sell or buy for the underlying asset.
		 * Note we deal with this in the vertex created by addVertex(stringForStrategyValue..)
		 */
		.addLayer(stringForStrategy,new DenseLayer.Builder().nIn(numberOfNodesForSecondLayer).nOut(1).activation(new ActivationIdentity()).build(), stringForSecondHiddenLayer);

		String stringForOldPrice = "price" + 0;

		/*
		 * Now we proceed with a for loop which runs over time in order to construct the rest of the sub-networks. Each sub-network will provide
		 * the optimal strategy at the specific time. In the meantime, we also have to take care of updating the cumulative value of
		 * the strategy, because this will contribute to the loss function which we need to train the network      
		 */
		for(int timeIndex = 1; timeIndex < numberOfTimes - 1; timeIndex ++) {

			/*
			 * All these strings identify the specific layer and vertices. Their name must be always different, which we achieve by placing the specific
			 * iteration number at the end
			 */
			stringForPrice = "price" + timeIndex;
			stringForFirstHiddenLayer = "firstHiddenLayer" + timeIndex;
			stringForSecondHiddenLayer = "secondHiddenLayer" + timeIndex;
			stringForStrategyValue = "strategyValue" + timeIndex;
			stringForCumulativeStrategyValue  = "cumulativeStrategyValue" + timeIndex;
			stringForIncrement = "increment" + timeIndex;

			//here we basically proceed as above, adding further inputs, vertices and layers to the network at each iteration

			builder.addInputs(stringForPrice)//we start from the new price at time t_{timeIndex}
			/*
			 * Note how a vertex is constructed: string which identifies this vertex, name of the operation that must be performed
			 * on the vertex to get the value of this vertex, and string which identifies the vertex on which the operation must be
			 * performed. So: in this case, based on the price, we compute the increment with respect to the last price.. 
			 */
			.addVertex(stringForIncrement, new ElementWiseVertex(Op.Subtract), stringForPrice, stringForOldPrice)
			//..and based on this increment, the profit and loss of the last computed strategy..
			.addVertex(stringForStrategyValue, new ElementWiseVertex(Op.Product), stringForIncrement, stringForStrategy)
			//..which we then sum to the vertex identified by stringForOldCumulativeStrategyValue, to compute the new value
			.addVertex(stringForCumulativeStrategyValue, new ElementWiseVertex(Op.Add), stringForStrategyValue, stringForOldCumulativeStrategyValue)
			//now we go back to price, and give it as an input to the first layer of the new sub-network
			.addLayer(stringForFirstHiddenLayer, new DenseLayer.Builder().nIn(1).nOut(numberOfNodesForFirstLayer)
					.activation(new ActivationReLU()).build(), stringForPrice)
			//and we proceed forward to the second layer
			.addLayer(stringForSecondHiddenLayer, new DenseLayer.Builder().nIn(numberOfNodesForFirstLayer)
					.nOut(numberOfNodesForSecondLayer).activation(new ActivationReLU()).build(), stringForFirstHiddenLayer);

			//we update the string for the strategy.. 
			stringForStrategy = "strategy" + timeIndex;

			//..and at a node identified by this string, we place the output of the second layer: it is the new computed value of the optimal strategy
			builder.addLayer(stringForStrategy,new DenseLayer.Builder().nIn(numberOfNodesForSecondLayer).nOut(1)
					.activation(new ActivationIdentity()).build(), stringForSecondHiddenLayer);			 

			//finally, we update the strings for the old price and the old cumulative value of the strategy
			stringForOldPrice = "price" + timeIndex;
			stringForOldCumulativeStrategyValue  = "cumulativeStrategyValue" + timeIndex;
		}

		//now we are at the last time: the last strategy which was possible to compute was at the time before, now we just observe what we got

		builder.addInputs("lastPrice")//here is the last price

		//and based on this, as we did in the for loop, we compute increment, profit and loss for the last strategy and cumulative value
		.addVertex("lastIncrement", new ElementWiseVertex(Op.Subtract), "lastPrice", stringForPrice)
		.addVertex("lastStrategyValue", new ElementWiseVertex(Op.Product), "lastIncrement", stringForStrategy)
		.addVertex("lastCumulativeStrategyValue", new ElementWiseVertex(Op.Add), "lastStrategyValue", stringForOldCumulativeStrategyValue)
		//this is a dummy input: it must be zero, and we need it to compute the maximum between zero and underlying minus strike in the payoff
		.addInputs("zero")
		//a ShiftVertex which contains the difference between the value of last price and strike
		.addVertex("difference", new ShiftVertex(- strike), "lastPrice")
		//and then we take the positive part: this is the payoff
		.addVertex("payoff", new ElementWiseVertex(Op.Max), "difference", "zero")
		//this is the cumulative value of the strategy minus the payoff..
		.addVertex("strategyMinusPayoff",  new ElementWiseVertex(Op.Subtract), "lastCumulativeStrategyValue", "payoff")		
		//..and this is the final portfolio value: we just add the price at which the option has been sold
		.addVertex("finalPortfolioValue", new ShiftVertex(+ optionPrice), "strategyMinusPayoff")	

		/*
		 * This is now the output of our network, based on which we compute the loss function. Basically we want it to have the same
		 * value as finalPortfolioValue, but to be defined as a layer because in this way we can specify the loss function. It has
		 * one input and one output. Later, we force it its parameters to be one and zero, respectively, so that it has the value
		 * of  finalPortfolioValue.
		 */
		.addLayer("finalOutput", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
				.nIn(1).nOut(1).activation(new ActivationIdentity()).constrainAllParameters().build(), "finalPortfolioValue")
		.setOutputs("finalOutput");

		//little technical thing: how to construct a true network out of a configuration

		ComputationGraphConfiguration configuration = builder.build();

		network = new ComputationGraph(configuration);
		network.init();


		//we give the values of the parameters of the last layer (see above)
		Layer finalPortfolioOutputLayer = network.getLayer("finalOutput");

		// we get its parameters
		INDArray weights = finalPortfolioOutputLayer.getParam("W");
		INDArray biases = finalPortfolioOutputLayer.getParam("b");

		// and set the weights to one and biases to zero
		weights.assign(1.0);
		biases.assign(0.0);

	}


	// In this method, we create an object of type MultiDataSet and we use it to train the network 
	private void trainNetwork() throws CalculationException {

		/*
		 * The length of this array must be equal to the number of inputs that we give to the network constructed in the method above:
		 * we had one input at the beginning (equal to zero, for the initial value of the cumulative sum of the strategies' values),
		 * numberOfTimes prices and then the other dummy input equal to zero at the end
		 */
		INDArray[] inputsArrayTrain = new INDArray[numberOfTimes + 2];

		/*
		 * This dummy matrix of one column is given to construct inputsArrayTrain[0]: all its elements are equal to zero.
		 * The reason why we construct it as a matrix is that the constructor of NDArray takes matrices, and not one-dimensional
		 * arrays
		 */
		double[][] initialValueStrategiesTrain = new double[numberOfSimulations][1];

		inputsArrayTrain[0] = new NDArray(initialValueStrategiesTrain); 

		//in this for loop, we get the prices for every simulation at all times, and we use them to construct the NDArray objects
		for(int timeIndex = 0; timeIndex < numberOfTimes; timeIndex ++) {

			//they are kind of dummy matrices of one column: the constructor of NDArray accepts matrices, not one-dimensional arrays
			double[][] pricesAtTimeToTrain = new double[numberOfSimulations][1];

			//we fill all rows with the corresponding prices..
			for(int simulationIndex = 0; simulationIndex < numberOfSimulations; simulationIndex ++) {
				pricesAtTimeToTrain[simulationIndex][0] = pricesGenerator.getAssetValue(timeIndex, 0).get(simulationIndex);
			}
			//..and give the matrix to the constructor of NDArray
			inputsArrayTrain[timeIndex + 1] = new NDArray(pricesAtTimeToTrain);
		}

		//dummy matrix, same thing as for initialValueStrategiesTrain
		double[][] zerosTrain = new double[numberOfSimulations][1];

		inputsArrayTrain[numberOfTimes + 1] = new NDArray(zerosTrain);

		/*
		 * These are the outputs we would like to have: we would like all values of profits and losses to be zero: so, these are
		 * the values the network is trained to reach
		 */
		double[][] finalValuesTrain = new double[numberOfSimulations][1];

		//we express them in terms of NDArray..
		INDArray[] outputsArrayTrain = new INDArray[1];

		outputsArrayTrain[0] = new NDArray(finalValuesTrain);		

		//..and we construct a MultiDataSet out of the inputs and the outputs
		MultiDataSet dataSetTrain = new MultiDataSet(inputsArrayTrain, outputsArrayTrain);

		//we then train the network based on this data
		for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex ++) {
			network.fit(dataSetTrain);
		}
	}

	/**
	 * This method returns the final portfolio values (profits and losses only coming from the strategies minus payoff to pay plus price
	 * at which the derivative is sold) for simulated prices not necessarily part of the training set. These values are the outputs of
	 * the trained network.
	 *  
	 * @param numberOfSimulationsForTesting, the number of paths we want to simulate
	 * @param seedForTesting, the seed for the generator of the paths
	 * @return an array of doubles containing the final portfolio value for every path
	 * @throws CalculationException
	 */
	public double[] getPortfolioValuesForTesting(int numberOfSimulationsForTesting, int seedForTesting) throws CalculationException {

		//same thing as usual: we call the core private methods only once
		if (network == null) {
			constructNetwork();
			trainNetwork();
		}

		/*
		 * Here we specify that we want a AssetModelMonteCarloSimulationModel object which is the same as pricesGenerator but:
		 * - with different seed
		 * - with different number of simulations
		 */

		//here we specify the different seed..
		AssetModelMonteCarloSimulationModel generatorWithModifiedSeed = pricesGenerator.getCloneWithModifiedSeed(seedForTesting);

		//..and here that we want to change the number of simulations, and how many we want now
		final Map<String, Object> mapToChangeTheNumberOfSimulations = new HashMap<String, Object>();
		mapToChangeTheNumberOfSimulations.put("numberOfSimulations", numberOfSimulationsForTesting);

		//we construct the generator..
		AssetModelMonteCarloSimulationModel testingPricesGenerator = generatorWithModifiedSeed.getCloneWithModifiedData(mapToChangeTheNumberOfSimulations);

		//..and then the inputs to be given to the generator. We do that exactly in the same way as for constructing inputsArrayTrain in the method above 
		INDArray[] inputsArrayTest = new INDArray[numberOfTimes + 2];

		double[][] initialValueStrategiesTest = new double[numberOfSimulationsForTesting][1];

		inputsArrayTest[0] = new NDArray(initialValueStrategiesTest); 


		for(int timeIndex = 0; timeIndex < numberOfTimes; timeIndex ++) {
			double[][] pricesAtTimeToTest = new double[numberOfSimulationsForTesting][1];

			for(int simulationIndex = 0; simulationIndex < numberOfSimulationsForTesting; simulationIndex ++) {
				pricesAtTimeToTest[simulationIndex][0] = testingPricesGenerator.getAssetValue(timeIndex, 0).get(simulationIndex);
			}

			inputsArrayTest[timeIndex + 1] = new NDArray(pricesAtTimeToTest);
		}

		double[][] zerosTest = new double[numberOfSimulationsForTesting][1];

		inputsArrayTest[numberOfTimes + 1] = new NDArray(zerosTest);

		/*
		 * We then give inputsArrayTest to the output method of network: this method returns a matrix whose i-th
		 * row contains the values of the i-th output in the final layer of the network for all data.
		 * In our case, we only have one output, the price. So the matrix only has one column.
		 */
		INDArray[] portfolioValuesTestAsINDArray = network.output(inputsArrayTest);

		//we convert this column in an array of doubles and get what we want to return 
		double[] portfolioValuesTest = portfolioValuesTestAsINDArray[0].toDoubleVector();


		return portfolioValuesTest;
	}

}
