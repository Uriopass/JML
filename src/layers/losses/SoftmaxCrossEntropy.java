package layers.losses;

import math.Activations;
import math.Matrix;

public class SoftmaxCrossEntropy extends Loss {
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		loss = 0;
		return Activations.softmax(in, Matrix.AXIS_HEIGHT);
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		loss = 0;
		for (int ref = 0; ref < refs.width; ref++) {
			int correct_ref = refs.get_column(ref).argmax();
			
			loss += -Math.log(dout.v[correct_ref][ref]);
			dout.v[correct_ref][ref] -= 1;
		}
		loss /=  refs.width;
		return dout;
	}
	
	@Override
	public String toString() {
		return "SoftmaxLayer()";
	}
}
