package layers;

import math.FeatureMatrix;

public interface FeatureLayer {
	/**
	 * @param in input datas
	 * @param training indicates wether or not to retain information for backprop
	 * @return out output datas
	 */
	public abstract FeatureMatrix forward(FeatureMatrix in, boolean training);
	/**
	 * @param dout derivative with respect to out
	 * @return dout derivative with respect to in
	 */
	public abstract FeatureMatrix backward(FeatureMatrix dout);
	public abstract void apply_gradient();
	/*
	public abstract void write(PrintWriter pw);
	public abstract void read (Scanner sc);
	*/
}
