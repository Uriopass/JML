package layers.flat;

import layers.FlatLayer;
import math.Matrix;
import math.RandomGenerator;

/**
 * Couche de régularisation Dropout, permettant de réduire l'overfitting en augmentant la diversité des neurones.
 * Le code n'est pas commenté car cette couche est hors-programme.
 */
public class DropoutLayer implements FlatLayer {
	Matrix cache;
	public double keep_prob;
	public double scale;

	public DropoutLayer(double prob) {
		keep_prob = 1 - prob;
		scale = 1 / keep_prob;
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		if (training) {
			cache = new Matrix(in.width, in.height);
			for (int i = 0; i < cache.height; i++) {
				for (int j = 0; j < cache.width; j++) {
					cache.v[i][j] = ((RandomGenerator.uniform(0, 1) < keep_prob) ? scale : 0);
				}
			}
			in.hadamart(cache);
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		return dout.hadamart(cache);
	}

	@Override
	public String toString() {
		return "DropoutLayer(" + (1 - keep_prob) + ")";
	}
}
