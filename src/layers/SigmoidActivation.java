package layers;

import math.Matrix;

public class SigmoidActivation extends Layer {
	Matrix cache;
	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = 1/(1+Math.exp(-in.v[i][j]));
			}
		}
		cache = in;
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		for(int i = 0 ; i < dout.height ; i++) {
			for(int j = 0 ; j < dout.width ; j++) {
				dout.v[i][j] *= cache.v[i][j]*(1-cache.v[i][j]);
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {};
	
	@Override
	public String toString() {
		return "SigmoidActivation()";
	}
}
