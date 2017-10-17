package layers;

import math.Matrix;
import math.Vector;

public class BatchnormLayer extends Layer {
	
	final static double epsilon = 1e-5;
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		// Step 1
		Vector mean = in.sum(Matrix.AXIS_WIDTH).scale(-1.0 / in.width);
		
		// Step 2
		in.add(mean, Matrix.AXIS_WIDTH);
		Matrix var_mini = new Matrix(in);
		// Step 3
		var_mini.hadamart(var_mini);
		
		// Step 4
		Vector var = var_mini.sum(Matrix.AXIS_WIDTH);
		
		// Step 5
		var.add(epsilon);
		
		return null;
	}

	@Override
	public Matrix backward(Matrix dout) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void apply_gradient() {
		// TODO Auto-generated method stub
		
	}

}
