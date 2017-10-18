package layers;

import math.Matrix;

public class ImageQuadraticLoss extends Layer {
	
	Matrix ref;
	public double loss;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		return in;
	}

	public void feed_ref(Matrix ref) {
		this.ref = ref;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		dout.add(ref.scale(-1));
		loss = 0.5*Matrix.hadamart(dout, dout).sum()/dout.width;
		return dout;
	}

	@Override
	public void apply_gradient() {
	}
	
	@Override
	public String toString() {
		return "ImageQuadraticLoss()";
	}
	
}
