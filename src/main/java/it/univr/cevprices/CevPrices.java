package it.univr.cevprices;


import net.finmath.functions.NonCentralChiSquaredDistribution;

/**
 * This class has two methods to compute prices for call options when the underlying is the CEV process  
 * 
 * @author Andrea Mazzon
 *
 */
public class CevPrices {


	/**
	 * It gives the price of the call when the exponent is strictly bigger than one
	 * @param initialValue
	 * @param sigma
	 * @param exponent: it must be strictly bigger than one
	 * @param maturity
	 * @param strike
	 * @return the price of the call when the exponent is strictly bigger than one
	 */
    public static double CEVPriceCallForExponentBiggerThanOne(double initialValue, double sigma, double exponent, double maturity, double strike) {
        double nu = 1 / (2 * (exponent - 1));
        double delta = (1 - 2 * exponent) / (1 - exponent);
        double transformedStrike = Math.pow(strike, 2 * (1 - exponent)) / (Math.pow(sigma * (1 - exponent), 2));
        double transformedInitialValue = Math.pow(initialValue, 2 * (1 - exponent)) / (Math.pow(sigma * (1 - exponent), 2));
        NonCentralChiSquaredDistribution chiSquaredDist1 = new NonCentralChiSquaredDistribution(2 * nu, transformedStrike / maturity);
        NonCentralChiSquaredDistribution chiSquaredDist2 = new NonCentralChiSquaredDistribution(delta, transformedInitialValue / maturity);
        double firstChiSquared = chiSquaredDist1.cumulativeDistribution(transformedInitialValue / maturity);
        double secondChiSquared = chiSquaredDist2.cumulativeDistribution(transformedStrike / maturity);
        return initialValue * (1 - firstChiSquared) - strike * secondChiSquared;
    }

    
    /**
	 * It gives the price of the call when the exponent is smaller or equal one
	 * @param initialValue
	 * @param sigma
	 * @param exponent: it must be smaller or equal one
	 * @param maturity
	 * @param strike
	 * @return the price of the call when the exponent is smaller or equal one
	 */
    public static double CEVPriceCallForExponentSmallerEqualOne(double initialValue, double sigma, double exponent, double T, double strike) {
        double delta = (1 - 2 * exponent) / (1 - exponent);
        double transformedStrike = Math.pow(strike, 2 * (1 - exponent)) / (Math.pow(sigma, 2) * Math.pow((1 - exponent), 2));
        double transformedInitialValue = Math.pow(initialValue, 2 * (1 - exponent)) / (Math.pow(sigma, 2) * Math.pow((1 - exponent), 2));
        NonCentralChiSquaredDistribution chiSquaredDist1 = new NonCentralChiSquaredDistribution(4 - delta, transformedStrike / T);
        NonCentralChiSquaredDistribution chiSquaredDist2 = new NonCentralChiSquaredDistribution(2 - delta, transformedInitialValue / T);
        double firstChiSquared = chiSquaredDist1.cumulativeDistribution(transformedStrike / T);
        double secondChiSquared = chiSquaredDist2.cumulativeDistribution(transformedInitialValue / T);
        return initialValue * (1 - firstChiSquared) - strike * secondChiSquared;
    }


}
