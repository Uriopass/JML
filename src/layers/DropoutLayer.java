package layers;

import java.util.Random;

import math.Matrix;
import math.RandomGenerator;

public class DropoutLayer extends Layer {
	Matrix cache;
	public double keep_prob;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training) {
			cache = new Matrix(in.width, in.height);
			for(int i = 0 ; i < cache.height ; i++) {
				for(int j = 0 ; j < cache.width ; j++) {
					cache.v[i][j] = (RandomGenerator.uniform(0, 1) > keep_prob)?1:0;
				}
			}
			in.hadamart(cache);	
		} else {
			in.scale(1.0 / keep_prob);
		}
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix backward(Matrix dout) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void apply_gradient() {
		
	}

}
