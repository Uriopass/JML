package layers.flat;

import layers.FlatLayer;
import math.Matrix;
import math.RandomGenerator;

public class GaussianNoise implements FlatLayer {

	public double variance;
	public GaussianNoise(double variance) {
		this.variance = variance;
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] += RandomGenerator.gaussian(variance);
			}
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		return dout;
	}

	@Override
	public void apply_gradient() {
		// TODO Auto-generated method stub

	}

}
