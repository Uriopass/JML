package layers;

import math.Matrix;

public class EntropyLoss implements FlatLayer {
	
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
		loss = 0;
		for(int i = 0 ; i < dout.height ; i++) {
			for(int j = 0 ; j < dout.width ; j++) {
				double y = ref.v[i][j];
				double y_prime = dout.v[i][j];
				loss -= (1-y)*Math.log(Math.max(1e-10, 1-y_prime)) + y*Math.log(Math.max(1e-10, y_prime));
				dout.v[i][j] = (y-y_prime)/(y_prime-y_prime*y_prime);
			}
		}
		loss /= dout.width;
		
		return dout;
	}

	@Override
	public void apply_gradient() {
	}
	
	@Override
	public String toString() {
		return "ImageEntropyLoss()";
	}
	
}
