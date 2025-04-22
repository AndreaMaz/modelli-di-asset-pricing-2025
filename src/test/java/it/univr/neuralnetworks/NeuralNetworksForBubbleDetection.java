package it.univr.neuralnetworks;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class NeuralNetworksForBubbleDetection {

	public static void main(String[] args) {
		
		double learningRate = 0.001;
		int ourSeed = 1897;
		
		int numberOfStrikes = 30;
		int numberOfMaturities = 50;
		
		int numberOfNodesFirstHiddenLayer = 200;
		int numberOfNodesSecondHiddenLayer = 100;
		
		MultiLayerConfiguration layersConstruction = new NeuralNetConfiguration.Builder()
				.updater(new Adam(learningRate))
				.seed(ourSeed)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(numberOfStrikes*numberOfMaturities)
						.nOut(numberOfNodesFirstHiddenLayer) 
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(numberOfNodesFirstHiddenLayer)
						.nOut(numberOfNodesSecondHiddenLayer) 
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder()
						.nIn(numberOfNodesSecondHiddenLayer)
						.nOut(1) 
						.activation(Activation.SIGMOID)
						.lossFunction(LossFunction.XENT)
						.build())
				.build();
		
		MultiLayerNetwork ourNetwork = new MultiLayerNetwork(layersConstruction);
		ourNetwork.init();
		
		double initialValueUnderlying = 2.0;
		
		double smallestValueStrike = 0.5;
		double biggestValueStrike = 10;
		
		double smallestValueMaturity = 0.1;
		double biggestValueMaturity = 1.5;
		
		double[] strikes = new double[numberOfStrikes];
		double[] maturities = new double[numberOfMaturities];
		
		double strikeStep = (biggestValueStrike-smallestValueStrike)/(numberOfStrikes-1);
		double maturityStep = (biggestValueMaturity-smallestValueMaturity)/(numberOfMaturities-1);
		
		for (int i = 0; i <= numberOfStrikes; i++) {
			strikes[i] = smallestValueStrike + i*strikeStep;
		}
		
		for (int j = 0; j <= numberOfMaturities; j++) {
			maturities[j] = smallestValueMaturity + j*maturityStep;
		}
		
	}

}
