package layers.activations;

import layers.FlatLayer;
import math.Matrix;

/**
 * Activation Tangeante Hyperbolique (TanH)
 * - TanH(x)  = (e^x - e^(-x))/(e^x + e^(-x))
 * - TanH'(x) = 1 - TanH(x)^2
 */
public class TanhActivation implements FlatLayer {
	public Matrix cache;

	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = Math.tanh(in.v[i][j]);
			}
		}
		if(training)
			cache = new Matrix(in);
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		for(int i = 0 ; i < dout.height ; i++) {
			for (int j = 0; j < dout.width; j++) {
				dout.v[i][j] *= 1 - cache.v[i][j]*cache.v[i][j];
			}
		}
		return dout;
	}
	@Override
	public String toString() {
		return "TanhActivation()";
	}
}
