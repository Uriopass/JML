package layers.activations;

import layers.FlatLayer;
import math.Matrix;

/**
 * Activation Sigmoid
 * - sigmoid(x)  = 1 / (1 + e^(-x))
 * - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 */
public class SigmoidActivation implements FlatLayer {
	public Matrix cache;

	@Override
	public String toString() {
		return "SigmoidActivation()";
	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = 1 / (1 + Math.exp(-in.v[i][j]));
			}
		}
		if(training) {
			cache = new Matrix(in);
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		for(int i = 0 ; i < dout.height ; i++) {
			for (int j = 0; j < dout.width; j++) {
				dout.v[i][j] *= cache.v[i][j]*(1-cache.v[i][j]);
			}
		}
		return dout;
	}
}
