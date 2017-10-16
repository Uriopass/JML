package layers;

import math.Matrix;

public abstract class Layer {
	/**
	 * @param in input datas
	 * @param training indicates wether or not to retain information for backprop
	 * @return out output datas
	 */
	public abstract Matrix forward(Matrix in, boolean training);
	/**
	 * @param dout derivative with respect to out
	 * @return dout derivative with respect to in
	 */
	public abstract Matrix backward(Matrix dout);
	public abstract void apply_gradient();
}
