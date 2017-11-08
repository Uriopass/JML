package layers.activations;

import layers.FeatureLayer;
import layers.FlatLayer;
import math.FeatureMatrix;
import math.Matrix;
import math.Vector;

public abstract class ActivationLayer implements FlatLayer, FeatureLayer {
	protected boolean needs_cache_before = false;
	protected boolean needs_cache_after  = false;
	
	public abstract double activation_forward(double in);
	public abstract double activation_backward();

	private Matrix m_cache_before;
	private Matrix m_cache_after;
	
	private Vector v_cache_before;
	private Vector v_cache_after;

	private FeatureMatrix f_cache_before;
	private FeatureMatrix f_cache_after;
	
	private int i, j, f;
	
	protected double get_before() {
		if(m_cache_before != null)
			return m_cache_before.v[i][j];
		if(f_cache_before != null)
			return f_cache_before.v[f].v[i][j];
		if(v_cache_before != null)
			return v_cache_before.v[i];
		throw new RuntimeException("Asking for before, but no before found");
	}
	
	protected double get_after() {
		if(m_cache_after != null)
			return m_cache_after.v[i][j];
		if(f_cache_after != null)
			return f_cache_after.v[f].v[i][j];
		if(v_cache_after != null)
			return v_cache_after.v[i];
		throw new RuntimeException("Asking for after, but no after found");
	}
	
	
	@Override
	public Matrix forward(Matrix in, boolean training) {
		if(training && needs_cache_before)
			m_cache_before = new Matrix(in);
		for(int i = 0 ; i < in.height ; i++) {
			for(int j = 0 ; j < in.width ; j++) {
				in.v[i][j] = activation_forward(in.v[i][j]);
			}
		}
		if(training && needs_cache_after)
			m_cache_after = new Matrix(in);
		return in;
	}
	
	public Vector forward(Vector in, boolean training) {
		if(training && needs_cache_before)
			v_cache_before = new Vector(in);
		for(int i = 0 ; i < in.length ; i++) {
			in.v[i] = activation_forward(in.v[i]);
		}
		if(training && needs_cache_after)
			v_cache_after = new Vector(in);
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
	
	public Vector backward(Vector dout) {
		for(int i = 0 ; i < dout.length ; i++) {
			this.i = i;
			dout.v[i] *= activation_backward();
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
