package layers.losses;

import math.Matrix;

public class QuadraticLoss extends Loss {
	
	Matrix ref;
	public double loss;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		return in;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		dout.add(ref.scale(-1));
		//dout.print_values();
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
