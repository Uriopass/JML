package math;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
/**
 * Générateur aléatoire déterministe si initialisé avec un seed, effectivement une sourcouche de java.util.Random
 */
public class RandomGenerator {
	public static Random r;

	public static void init(long seed) {
		r = new Random(seed);
	}

	public static Matrix normal(double mean, double variance, int width, int height) {
		if (r == null)
			r = new Random();
		Matrix m = new Matrix(width, height);
		for (int i = 0; i < height ; i++) {
			for (int j = 0; j < width ; j++) {
				m.v[i][j] = r.nextGaussian()*variance + mean;
			}
		}
		return m;
	}
	
	/**
	 * Renvoie un réel au hasard dans une distribution gaussienne avec une variance donnée
	 */
	public static double gaussian(double variance) {
		if (r == null)
			r = new Random();
		return r.nextGaussian() * variance;
	}

	/**
	 * Renvoie un réel au hasard de manière uniforme dans l'intervalle [min ; max]
	 */
	public static double uniform(double min, double max) {
		if (r == null)
			r = new Random();
		return r.nextDouble() * (max - min) + min;
	}

	/**
	 * Renvoie un entier au hasard dans l'intervalle [min ; max[
	 */
	public static int uniform_int(int inclusive_min, int exclusive_max) {
		if (r == null)
			r = new Random();
		return r.nextInt(exclusive_max - inclusive_min) - inclusive_min;
	}

	/**
	 * Renvoie un entier au hasard dans l'intervalle [0 ; max[
	 */
	public static int uniform_int(int exclusive_max) {
		return uniform_int(0, exclusive_max);
	}

	/**
	 * Renvoie un échantillon de count nombre parmis bound entiers
	 * @param bound Nombre d'entier pour la population, par exemple 3 donne [0; 1; 2]
	 * @param count Taille de l'échantillon
	 */
	public static int[] sample(int bound, int count) {
		if (r == null)
			r = new Random();
		if (count > bound) {
			throw new RuntimeException("Asking too much numbers without replace");
		}
		ArrayList<Integer> integers = new ArrayList<Integer>();
		for (int i = 0; i < bound; i++) {
			integers.add(i);
		}
		Collections.shuffle(integers);
		int[] sample = new int[count];
		for (int i = 0; i < count; i++) {
			sample[i] = integers.get(i);
		}
		return sample;
	}

}
