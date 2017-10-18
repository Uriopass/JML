package layers.activations;

import layers.FeatureLayer;
import layers.Layer;
import math.FeatureMatrix;
import math.Matrix;

public abstract class ActivationLayer implements Layer, FeatureLayer {
	protected boolean needs_cache_before = false;
	protected boolean needs_cache_after  = true;
	protected boolean is_feature_layer   = false;
	
	public abstract double activation_forward(double in);
	public abstract double activation_backward();

	private Matrix cache_before;
	private Matrix cache_after;

	private FeatureMatrix f_cache_before;
	private FeatureMatrix f_cache_after;
	
	private int i, j, f;
	
	protected double get_before() {
		if(is_feature_layer)
			return f_cache_before.v[f].v[i][j];
		return cache_before.v[i][j];
	}
	
	protected double get_after() {
		if(is_feature_layer)
			return f_cache_after.v[f].v[i][j];
		return cache_after.v[i][j];
	}
	
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training && needs_cache_before)
			cache_before = new Matrix(in);
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = activation_forward(in.v[i][j]);
			}
		}
		if(training && needs_cache_after)
			cache_after = new Matrix(in);
		return in;
	}
	
	public FeatureMatrix forward(FeatureMatrix in, boolean training) {
		if(training && needs_cache_before)
			f_cache_after = new FeatureMatrix(in);
		for(int f = 0 ; f < in.features ; f++) {
			for(int i = 0 ; i < in.height ; i++) {
				for(int j = 0 ; j < in.width ; j++) {
					in.v[f].v[i][j] = activation_forward(in.v[f].v[i][j]);
				}
			}
		}
		if(training && needs_cache_after)
			f_cache_after = new FeatureMatrix(in);
		return in;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		for(int i = 0 ; i < dout.height ; i++) {
			this.i = i;
			for(int j = 0 ; j < dout.width ; j++) {
				this.j = j;
				dout.v[i][j] *= activation_backward();
			}
		}
		return dout;
	}
	
	public FeatureMatrix backward(FeatureMatrix dout) {
		for (int f = 0; f < dout.features; f++) {
			this.f = f;
			for(int i = 0 ; i < dout.height ; i++) {
				this.i = i;
				for(int j = 0 ; j < dout.width ; j++) {
					this.j = j;
					dout.v[f].v[i][j] *= activation_backward();
				}
			}
		}
		return dout;
	}

	@Override
	public void apply_gradient() {};
}
