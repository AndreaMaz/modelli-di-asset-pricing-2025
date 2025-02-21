package it.univr.usefulmethodsarrays;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.numbers.core.Precision;

public class UsefulMethodsArrays {

	/**
	 * It returns the biggest element of a one-dimensional array of doubles
	 *
	 * @param vector the one-dimensional array
	 * @return the biggest element of the one-dimensional array
	 */
	public static double getMax(double[] array) {
		double maximum = Arrays.stream(array).max().getAsDouble();
		return maximum;
	}

	
	/**
	 * It computes and returns the maximum absolute difference between two arrays
	 * @param firstArray
	 * @param secondArray
	 * @return the maximum absolute difference between firstArray and secondArray
	 * @throws Exception if the two arrays have different lengths 
	 */
	public static double getMaxDifference(double[] firstArray, double[] secondArray) throws Exception {

		if (firstArray.length != secondArray.length) {
			throw new Exception();
		}

		return IntStream.range(0, firstArray.length)//the indices range from 0 to array.length-1
				.mapToDouble(i -> Math.abs(firstArray[i] - secondArray[i])).max().getAsDouble();
	}

	/**
	 * It returns the index which maximizes the value of an array, up to precision of two digits. In case of more elements
	 * of the array achieving the maximum, the smallest index is returned
	 * @param array
	 * @return the index which maximizes the value of an array
	 */
	public static int getMaximizingIndex(double[] array) {
		double maximum = getMax(array);

		int[] maximizingIndices = IntStream.range(0, array.length).filter(i -> Precision.round(array[i],4) == Precision.round(maximum,4)).toArray();
		return maximizingIndices[0];
	}


	/**
	 * It solves the linear system Ax=b
	 * @param matrix, A
	 * @param knownVector, b
	 * @return the solution x
	 */
	public static double[] solveLinearSystem(double[][] matrix, double[] knownVector) {

		//we use here the implementation of org.apache.commons.math3.linear
		RealMatrix newMatrix = MatrixUtils.createRealMatrix(matrix);
		DecompositionSolver solver = new LUDecomposition(newMatrix).getSolver();
		RealVector constants = new ArrayRealVector(knownVector, false);
		RealVector solution = solver.solve(constants);
		return solution.toArray();
	}
	
	
	/**
	 * It computes and returns the maximum sum of the elements of the rows of the matrix which is the difference between
	 * the two matrices given in input, divided by the number of columns of the matrices.
	 * @param firstMatrix
	 * @param secondMatrix
	 * @return the maximum sum of the elements of the rows of the matrix which is the difference between
	 * 			the two matrices given in input, divided by the number of columns of the matrices
	 */
	public static double getNormDifference(double[][] firstMatrix, double[][] secondMatrix) throws Exception {

		RealMatrix newFirstMatrix = MatrixUtils.createRealMatrix(firstMatrix);
		RealMatrix newSecondMatrix = MatrixUtils.createRealMatrix(secondMatrix);

		int numberOfColumns = newFirstMatrix.getColumnDimension();

		return newFirstMatrix.subtract(newSecondMatrix).getNorm()/numberOfColumns;
	}
	
}
