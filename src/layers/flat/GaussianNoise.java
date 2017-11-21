package layers.flat;

import layers.FlatLayer;
import math.Matrix;
import math.RandomGenerator;

/**
 * Bruit gaussien par addition.
 * y = x + G(variance)
 */
public class GaussianNoise implements FlatLayer {

	public double variance;

	/**
	 * @param variance Variance du bruit utilisé
	 */
	public GaussianNoise(double variance) {
		this.variance = variance;
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		for (int i = 0; i < in.height; i++) {
			for (int j = 0; j < in.width; j++) {
				in.v[i][j] += RandomGenerator.gaussian(variance);
			}
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		return dout;
	}
}
