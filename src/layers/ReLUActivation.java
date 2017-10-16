package layers;

import math.Matrix;

public class ReLUActivation extends Layer {
	Matrix cache;
	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = Math.max(0, in.v[i][j]);
			}
		}
		cache = in;
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		for(int i = 0 ; i < dout.height ; i++) {
			for(int j = 0 ; j < dout.width ; j++) {
				int sig = (int) Math.signum(cache.v[i][j]);
				dout.v[i][j] *= (sig+1)/2;
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {};
	s
	@Override
	public String toString() {
		return "TanhActivation()";
	}
}
