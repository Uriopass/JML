package layers.losses;

import math.Activations;
import math.Matrix;

/**
 * Applies a sigmoid activation then uses a binary (entropy) loss on the output
 */
public class SigmoidBinaryEntropyLoss extends Loss {
	@Override
	public Matrix forward(Matrix in, boolean training) {
		for(int i = 0 ; i < in.height ; i++) {
			for (int j = 0; j < in.width ; j++) {
				in.v[i][j] = Activations.sigmoid(in.v[i][j]);
			}
		}
		return in;
	}

	@Override
	public Matrix backward(Matrix dout, boolean train) {
		loss = 0;
		for (int i = 0; i < dout.height; i++) {
			for (int j = 0; j < dout.width; j++) {
				double y = refs.v[i][j];
				double y_prime = dout.v[i][j];
				if(y == 0) {
					loss -= Math.log(1 - y_prime);
				} else if(y == 1) {
					loss -= Math.log(y_prime);
				}
			}
		}
		loss /= dout.width;
		dout.add(refs.scale(-1));
		return dout;
	}
	
}
