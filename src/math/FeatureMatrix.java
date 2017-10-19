package math;

public class FeatureMatrix {
	public Matrix[] v;
	public int features, width, height;
	
	public FeatureMatrix(int features, int width, int height) {
		v = new Matrix[features];
		this.features = features;
		this.width = width;
		this.height = height;
		
		for(int i = 0 ; i < features ; i++) {
			v[i] = new Matrix(width, height);
		}
	}

	public FeatureMatrix(FeatureMatrix in) {
		this.features = in.features;
		this.width = in.width;
		this.height = in.height;
	
		v = new Matrix[features];
	
		for(int i = 0 ; i < features ; i++) {
			v[i] = new Matrix(in.v[i]);
		}
	}
	
	public void zero_pad(int pad) {
		for(Matrix m : v)
		{
			m.zero_pad(pad);
		}
	}
	
	@Override
	public String toString() {
		return "("+features+", "+width+", "+height+")";
	}
}
