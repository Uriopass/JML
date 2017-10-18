package layers;

import math.Matrix;

public abstract class ActivationLayer extends Layer{
	public abstract double activation_forward(double in);
	public abstract double activation_backward(double in);
	
	Matrix cache;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = activation_forward(in.v[i][j]);
			}
		}
		if(training)
			cache = in;
		return in;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		for(int i = 0 ; i < dout.height ; i++) {
			for(int j = 0 ; j < dout.width ; j++) {
				dout.v[i][j] *= activation_backward(cache.v[i][j]);
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {};
}
