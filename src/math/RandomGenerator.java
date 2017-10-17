package math;

import java.util.Random;

public class RandomGenerator {
	public static Random r;
	public static void init(long seed) {
		r = new Random(seed);
	}
	
	public static double gaussian(double variance) {
		if(r == null)
			r = new Random();
		return r.nextGaussian()*variance;
	}
	
	public static double uniform(double min, double max) {
		if(r == null)
			r = new Random();
		return r.nextDouble()*(max-min)+min;
	}
}
