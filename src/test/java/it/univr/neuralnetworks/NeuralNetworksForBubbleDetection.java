package it.univr.neuralnetworks;


import java.util.Arrays;
import java.util.Random;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import it.univr.cevprices.CevPrices;

public class NeuralNetworksForBubbleDetection {

	public static void main(String[] args) {
		
		// FASE COSTRUZIONE NETWORK
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
		
		//FASE DEFINIZIONE TRAINING SET
		
		double initialValueUnderlying = 2.0;
		double sigmaUnderlying = 0.3;
		
		double smallestValueStrike = 0.5;
		double biggestValueStrike = 10;
		
		double smallestValueMaturity = 0.1;
		double biggestValueMaturity = 1.5;
		
		double[] strikes = new double[numberOfStrikes];
		double[] maturities = new double[numberOfMaturities];
		
		double strikeStep = (biggestValueStrike-smallestValueStrike)/(numberOfStrikes-1);
		double maturityStep = (biggestValueMaturity-smallestValueMaturity)/(numberOfMaturities-1);
		
		for (int i = 0; i < numberOfStrikes; i++) {
			strikes[i] = smallestValueStrike + i*strikeStep;
		}
		
		for (int j = 0; j < numberOfMaturities; j++) {
			maturities[j] = smallestValueMaturity + j*maturityStep;
		}
		
		int numberOfStrictLocalMartingalesForTraining = 100;
		int numberOfTrueMartingalesForTraining = 100;
		
		int numberOfRowsMatrixTraining = numberOfStrictLocalMartingalesForTraining+numberOfTrueMartingalesForTraining;
		
		double [][] pricesForTraining = 
				new double[numberOfRowsMatrixTraining][numberOfStrikes*numberOfMaturities];
		
		double smallestExponentTrueMartingales = 0.5;
		double biggestExponentTrueMartingales = 1;
		
		double smallestExponentStrictLocalMartingales = 1.01;
		double biggestExponentStrictLocalMartingales = 1.5;
		
		Random randomGenerator = new Random();
		
		double differenceExponentsStrict = biggestExponentStrictLocalMartingales-smallestExponentStrictLocalMartingales;
		
		//tutti beta_i > 1
		for (int i = 0; i < numberOfStrictLocalMartingalesForTraining; i++ ){
			double randomNumber = randomGenerator.nextDouble();
			double exponent = smallestExponentStrictLocalMartingales+randomNumber*differenceExponentsStrict;

			for (int maturityIndex = 0; maturityIndex<numberOfMaturities; maturityIndex++) {
				for (int strikeIndex = 0; strikeIndex<numberOfStrikes; strikeIndex++) {
					double callPrice =
							CevPrices.CEVPriceCallForExponentBiggerThanOne(initialValueUnderlying, sigmaUnderlying, exponent, maturities[maturityIndex], strikes[strikeIndex]);
					pricesForTraining[i][maturityIndex*numberOfStrikes+strikeIndex] = callPrice;     
				}
			}		
		}

		double differenceExponentsTrue = biggestExponentTrueMartingales-smallestExponentTrueMartingales;


		//tutti beta_i <= 1
		for (int i = numberOfStrictLocalMartingalesForTraining; i < numberOfRowsMatrixTraining; i++ ){
			double randomNumber = randomGenerator.nextDouble();
			double exponent = smallestExponentTrueMartingales+randomNumber*differenceExponentsTrue;

			for (int maturityIndex = 0; maturityIndex<numberOfMaturities; maturityIndex++) {
				for (int strikeIndex = 0; strikeIndex<numberOfStrikes; strikeIndex++) {
					double callPrice =
							CevPrices.CEVPriceCallForExponentSmallerEqualOne(initialValueUnderlying, sigmaUnderlying, exponent, maturities[maturityIndex], strikes[strikeIndex]);
					pricesForTraining[i][maturityIndex*numberOfStrikes+strikeIndex] = callPrice;     
				}
			}		
		}
		
		
		double[][] labelsForTraining = new double[numberOfRowsMatrixTraining][1];
		
		for (int i = 0; i < numberOfStrictLocalMartingalesForTraining; i++ ){
			labelsForTraining[i][0] = 1;
		}

		for (int i = numberOfStrictLocalMartingalesForTraining; i < numberOfRowsMatrixTraining; i++ ){
			labelsForTraining[i][0] = 0;
		}
		
		INDArray pricesDataForTraining = Nd4j.create(pricesForTraining);
		INDArray labelsDataForTraining = Nd4j.create(labelsForTraining);

		DataSet trainingData = new DataSet(pricesDataForTraining, labelsDataForTraining);

		//FASE ADDESTRAMENTO DEL NETWORK
		
		int numberOfEpochs = 30;
		
		for (int epochIndex = 0; epochIndex < numberOfEpochs; epochIndex++) {
			ourNetwork.fit(trainingData);
		}

		//FASE TEST
		
		int numberOfStrictLocalMartingalesForTesting = 10;
		int numberOfTrueMartingalesForTesting = 10;
		
		int numberOfRowsMatrixTesting = numberOfStrictLocalMartingalesForTesting+numberOfTrueMartingalesForTesting;
		
		double [][] pricesForTesting = 
				new double[numberOfRowsMatrixTesting][numberOfStrikes*numberOfMaturities];

		//tutti beta_i > 1
		for (int i = 0; i < numberOfStrictLocalMartingalesForTesting; i++ ){
			double randomNumber = randomGenerator.nextDouble();
			double exponent = smallestExponentStrictLocalMartingales+randomNumber*differenceExponentsStrict;

			for (int maturityIndex = 0; maturityIndex<numberOfMaturities; maturityIndex++) {
				for (int strikeIndex = 0; strikeIndex<numberOfStrikes; strikeIndex++) {
					double callPrice =
							CevPrices.CEVPriceCallForExponentBiggerThanOne(initialValueUnderlying, sigmaUnderlying, exponent, maturities[maturityIndex], strikes[strikeIndex]);
					pricesForTesting[i][maturityIndex*numberOfStrikes+strikeIndex] = callPrice;     
				}
			}		
		}

		//tutti beta_i <= 1
		for (int i = numberOfStrictLocalMartingalesForTesting; i < numberOfRowsMatrixTesting; i++ ){
			double randomNumber = randomGenerator.nextDouble();
			double exponent = smallestExponentTrueMartingales+randomNumber*differenceExponentsTrue;

			for (int maturityIndex = 0; maturityIndex<numberOfMaturities; maturityIndex++) {
				for (int strikeIndex = 0; strikeIndex<numberOfStrikes; strikeIndex++) {
					double callPrice =
							CevPrices.CEVPriceCallForExponentSmallerEqualOne(initialValueUnderlying, sigmaUnderlying, exponent, maturities[maturityIndex], strikes[strikeIndex]);
					pricesForTesting[i][maturityIndex*numberOfStrikes+strikeIndex] = callPrice;     
				}
			}		
		}
		
		INDArray pricesDataForTesting = Nd4j.create(pricesForTesting);
		INDArray predictionsAsINDArray = ourNetwork.output(pricesDataForTesting);
		
		double[] predictions = predictionsAsINDArray.toDoubleVector();
		
		System.out.println("Probabilities outputs that underlyings are strict local martingales");
		System.out.println(Arrays.toString(predictions));
	}

}
