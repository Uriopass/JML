package layers.features;

import layers.FeatureLayer;
import math.FeatureMatrix;

public class MaxPooling implements FeatureLayer {

	public int stride;
	public int width_out, height_out;
	
	FeatureMatrix cache;
	
	public MaxPooling(int width_in, int height_in, int stride) {
		if(width_in%stride != 0) {
			throw new RuntimeException("Invalid stride for width "+width_in+" and stride "+stride);
		}
		if(height_in%stride != 0) {
			throw new RuntimeException("Invalid stride for height "+height_in+" and stride "+stride);
		}
		width_out = width_in/stride;
		height_out = height_in/stride;
		this.stride = stride;
	}
	
	@Override
	public FeatureMatrix forward(FeatureMatrix in, boolean training) {
		FeatureMatrix out = new FeatureMatrix(in.features, width_out, height_out);
		cache = new FeatureMatrix(in);
		for(int f = 0 ; f < in.features ; f++) {
			for(int i = 0 ; i < width_out ; i++) {
				for (int j = 0; j < height_out; j++) {
					double max = Float.NEGATIVE_INFINITY;
					for(int k = 0 ; k < stride ; k++) {
						for(int k2 = 0 ; k2 < stride ; k2++) {
							max = Math.max(max, in.v[f].v[i*stride+k][j*stride+k2]);
						}
					}
					out.v[f].v[i][j] = max;
					for(int k = 0 ; k < stride ; k++) {
						for(int k2 = 0 ; k2 < stride ; k2++) {
							cache.v[f].v[i*stride+k][j*stride+k2] = (in.v[f].v[i*stride+k][j*stride+k2]==max?1:0);
						}
					}
				}
			}
		}
		return out;
	}

	@Override
	public FeatureMatrix backward(FeatureMatrix dout) {
		for(int f = 0 ; f < cache.features ; f++) {
			for(int i = 0 ; i < cache.width ; i++) {
				for (int j = 0; j < cache.height; j++) {
					cache.v[f].v[i][j] *= dout.v[f].v[i/stride][j/stride];
				}
			}
		}
		return cache;
	}

	@Override
	public void apply_gradient() {
	}

	@Override
	public String toString() {
		return "MaxPooling("+stride+")";
	}
}
