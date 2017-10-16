package layers;

import math.Activations;
import math.Matrix;

public class SoftmaxLayer extends Layer {
	
	int[] refs;
	public int correct;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		correct = 0;
		return Activations.softmax(in, 1);
	}
	
	public void feedrefs(int[] refs) {
		this.refs = refs;
	}

	@Override
	public Matrix backward(Matrix dout) {
		correct = 0;
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
