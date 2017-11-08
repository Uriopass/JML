package layers.losses;

import math.Matrix;

public class EntropyLoss extends Loss {
	@Override
	public Matrix forward(Matrix in, boolean training) {
		return in;
	}
	
	@Override
	public Matrix backward(Matrix dout) {
		loss = 0;
		for(int i = 0 ; i < dout.height ; i++) {
			for(int j = 0 ; j < dout.width ; j++) {
				double y = refs.v[i][j];
				double y_prime = dout.v[i][j];
				loss -= (1-y)*Math.log(Math.max(1e-10, 1-y_prime)) + y*Math.log(Math.max(1e-10, y_prime));
				dout.v[i][j] = (y-y_prime)/(y_prime-y_prime*y_prime);
			}
		}
		loss /= dout.width;
		
		return dout;
	}
	
	@Override
	public String toString() {
		return "ImageEntropyLoss()";
	}
}
