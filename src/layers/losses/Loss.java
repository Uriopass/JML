package layers.losses;

import layers.FlatLayer;
import math.Matrix;

public abstract class Loss implements FlatLayer {
	Matrix refs;
	public double loss;
	
	public void feed_ref(Matrix ref) {
		this.refs = ref;
	}
	
	public static Matrix from_int_refs(int[] refs, int nb_class) {
		Matrix m = new Matrix(refs.length, nb_class);
		for(int i = 0 ; i < refs.length ; i++) {
			m.v[refs[i]][i] = 1;
		}
		return m;
	}
	
	@Override
	public void apply_gradient() {}
}
