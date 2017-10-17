package layers;

import math.Matrix;
import math.RandomGenerator;

public class DropoutLayer extends Layer {
	Matrix cache;
	public double keep_prob;
	public double scale;
	
	public DropoutLayer(double prob) {
		keep_prob = 1-prob;
		scale = 1/keep_prob;
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training) {
			cache = new Matrix(in.width, in.height);
			for(int i = 0 ; i < cache.height ; i++) {
				for(int j = 0 ; j < cache.width ; j++) {
					cache.v[i][j] = ((RandomGenerator.uniform(0, 1) < keep_prob)?scale:0);
				}
			}
			in.hadamart(cache);	
		} else {
			//in.scale(keep_prob);
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		return dout.hadamart(cache);
	}

	@Override
	public void apply_gradient() {}
	
	@Override
	public String toString() {
		return "DropoutLayer("+(1-keep_prob)+")";
	}

}
