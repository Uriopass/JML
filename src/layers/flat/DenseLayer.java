package layers.flat;

import layers.FlatLayer;
import layers.Parameters;
import layers.activations.ActivationLayer;
import layers.activations.ActivationParser;
import math.Matrix;

public class DenseLayer implements FlatLayer {
	public AffineLayer al;
	DropoutLayer dl;
	BatchnormLayer bl;
	ActivationLayer act;
	boolean dropout = false, batchnorm = false;

	public DenseLayer(int fan_in, int fan_out, double dropout, String activation, boolean batchnorm, Parameters p) {
		al = new AffineLayer(fan_in, fan_out, true, p);
		if (batchnorm) {
			bl = new BatchnormLayer(fan_out, p);
			batchnorm = true;
		}
		act = ActivationParser.getActivationByName(activation);
		if (dropout > 0) {
			dl = new DropoutLayer(dropout);
			this.dropout = true;
		}

	}

	@Override
	public Matrix forward(Matrix in, boolean training) {
		Matrix next = al.forward(in, training);
		if (batchnorm)
			next = bl.forward(next, training);
		if(act != null)
			next = act.forward(next, training);
		if (dropout)
			next = dl.forward(next, training);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		Matrix next = dout;
		if (dropout)
			next = dl.backward(next);
		if(act != null)
			next = act.backward(next);
		if (batchnorm)
			next = bl.backward(next);
		next = al.backward(next);
		return next;
	}

	@Override
	public void apply_gradient() {
		al.apply_gradient();
		if (batchnorm)
			bl.apply_gradient();
		if(act != null)
			act.apply_gradient();
		if (dropout)
			dl.apply_gradient();
	}

	@Override
	public String toString() {
		return al + "\n" + ((bl != null) ? bl + "\n" : "") + act + ((dl != null) ? "\n" + dl : "");
	}
}
