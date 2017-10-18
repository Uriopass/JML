package layers.featurelayers;

import layers.FeatureLayer;
import math.FeatureMatrix;
import math.Initialisations;
import math.Vector;

public class ConvolutionLayer implements FeatureLayer {
	public int width_out, height_out, features;
	public FeatureMatrix weights;
	public Vector biases;
	
	public ConvolutionLayer(int width_in, int height_in, int features_out, int conv_size, int stride, int padding) {
		int width_out = width_in-conv_size+2*padding;
		int height_out = width_in-conv_size+2*padding;
		if ((width_out%stride) != 0 || (height_out%stride) != 0) {
			throw new RuntimeException("Invalid convolution dimensions");
		}
		width_out = width_out/stride+1;
		height_out = height_out/stride+1;
		weights = new FeatureMatrix(features_out, conv_size, conv_size);
		biases = new Vector(features_out);
		for (int i = 0; i < weights.features; i++) {
			Initialisations.he_uniform(weights.v[i], conv_size*conv_size, 1);
		}
	}

	@Override
	public FeatureMatrix forward(FeatureMatrix in, boolean training) {
		
		return in;
	}

	@Override
	public FeatureMatrix backward(FeatureMatrix dout) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void apply_gradient() {
		// TODO Auto-generated method stub
		
	}
	
	
}
