package layers.flatlayers;

import layers.FlatLayer;
import layers.Parameters;
import layers.activations.ActivationLayer;
import layers.activations.ReLUActivation;
import layers.activations.SigmoidActivation;
import layers.activations.TanhActivation;
import math.Matrix;

public class DenseLayer implements FlatLayer {
	AffineLayer al;
	DropoutLayer dl;
	BatchnormLayer bl;
	ActivationLayer act;
	boolean dropout = false, batchnorm = false;
	
	
	public DenseLayer(int fan_in, int fan_out, double dropout, String activation, boolean batchnorm, Parameters p) {
		al = new AffineLayer(fan_in, fan_out, true, p);
		if(batchnorm) {
			bl = new BatchnormLayer(fan_out, p);
			batchnorm = true;
		}
		switch(activation.toLowerCase()) {
		case "tanh":
			act = new TanhActivation();
			break;
		case "sigmoid":
		case "sig":
			act = new SigmoidActivation();
			break;
		case "relu":
			act = new ReLUActivation();
			break;
			default:
				act = new TanhActivation();
		}
		if(dropout > 0) {
			dl = new DropoutLayer(dropout);
			this.dropout = true;
		}
		
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		Matrix next  = al.forward(in, training);
		if(batchnorm)
			next = bl.forward(next, training);
		next = act.forward(next, training);
		if(dropout)
			next = dl.forward(next, training);
		return next;
	}

	@Override
	public Matrix backward(Matrix dout) {
		Matrix next = dout;
		if(dropout)
			next = dl.backward(next);
		next = act.backward(next);
		if(batchnorm)
			next = bl.backward(next);
		next = al.backward(next);
		return next;
	}

	@Override
	public void apply_gradient() {
		al.apply_gradient();
		if(batchnorm)
			bl.apply_gradient();
		act.apply_gradient();
		if(dropout)
			dl.apply_gradient();
	}

	@Override
	public String toString() {
		return al+"\n"+((bl!=null)?bl+"\n":"")+act+((dl!=null)?"\n"+dl:"");
	}
}
