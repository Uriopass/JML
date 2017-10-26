package math;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class RandomGenerator {
	private static Random r;
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
	
	public static int uniform_int(int inclusive_min, int exclusive_max) {
		if(r == null)
			r = new Random();
		return r.nextInt(exclusive_max-inclusive_min)-inclusive_min;
	}
	
	public static int uniform_int(int exclusive_max) {
		return uniform_int(0, exclusive_max);
	}
	
	public static int[] sample(int bound, int count) {
		if(r == null)
			r = new Random();
		if(count > bound) {
			throw new RuntimeException("Asking too much numbers without replace");
		}
		ArrayList<Integer> integers = new ArrayList<Integer>();
		for(int i = 0 ; i < bound ; i++) {
			integers.add(i);
		}
		Collections.shuffle(integers);
		int[] sample = new int[count];
		for(int i = 0 ; i < count ; i++) {
			sample[i] = integers.get(i);
		}
		return sample;
	}
	
}
