package layers.flatlayers;

import layers.FlatLayer;
import math.Activations;
import math.Matrix;

public class SoftmaxCrossEntropy implements FlatLayer {
	
	int[] refs;
	public int correct;
	public double loss;
	@Override
	public Matrix forward(Matrix in, boolean training) {
		correct = 0;
		loss = 0;
		return Activations.softmax(in, 1);
	}
	
	public void feedrefs(int[] refs) {
		this.refs = refs;
	}

	@Override
	public Matrix backward(Matrix dout) {
		correct = 0;
		loss = 0;
		for (int ref = 0; ref < refs.length; ref++) {
			loss += -Math.log(dout.v[refs[ref]][ref]);
		}
		loss /=  refs.length;
		
		for (int i = 0; i < refs.length; i++) {
			if (dout.get_column(i).argmax() == refs[i]) {
				correct++;
			}
			dout.v[refs[i]][i] -= 1;
		}
		return dout;
	}

	@Override
	public void apply_gradient() {};

	@Override
	public String toString() {
		return "SoftmaxLayer()";
	}
}
