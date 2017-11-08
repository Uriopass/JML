package layers.activations;

import layers.FlatLayer;
import math.Activations;
import math.Matrix;
import math.Vector;

public class SoftmaxActivation implements FlatLayer {
	Matrix cache;
	@Override
	public Matrix forward(Matrix in, boolean training) {
		in = Activations.softmax(in, Matrix.AXIS_HEIGHT);
		if(training)
			cache = new Matrix(in);
		return in;
	}

	@Override
	public Matrix backward(Matrix dout) {
		for(int i = 0 ; i < dout.width ; i++) {
			Vector t = cache.get_column(i);
			double s = t.sum();
			for(int j = 0 ; j < dout.height ; j++) {
				double v = t.v[j];
				dout.v[j][i] *= v*(-s + 2*v - 1);
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {
		// TODO Auto-generated method stub
		
	}
}
