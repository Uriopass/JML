package layers.flat;

import layers.FlatLayer;
import math.Matrix;

public class ConstantLayer implements FlatLayer {
	public int width, height;
	double c;
	public ConstantLayer(int width, int height, double constant) {
		this.width = width;
		this.height = height;
		this.c = constant;
	}
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(c == 0)
			return new Matrix(width, height);
		return new Matrix(width, height).fill(c);
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		return dout;
	}
}
