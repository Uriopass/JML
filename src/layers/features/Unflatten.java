package layers.features;

import layers.Layer;
import math.FeatureMatrix;
import math.Matrix;

public class Unflatten implements Layer {
	public int features, width, height, outlength;
	Matrix cache;
	
	public Unflatten(int features, int width, int height) {
		this.features = features;
		this.width = width;
		this.height = height;
	}
	
	public FeatureMatrix forward(Matrix in, boolean training) {
		cache = in;
		FeatureMatrix result = new FeatureMatrix(features, width, height);
		for (int i = 0; i < features; i++) {
			for (int j = 0; j < height; j++) {
				for (int j2 = 0; j2 < width; j2++) {
					result.v[i].v[j][j2] = in.v[i*height*width+j*width+j2][0];
				}
			}
		}
		return result;
	}

	public Matrix backward(FeatureMatrix dout) {
		for (int i = 0; i < features; i++) {
			for (int j = 0; j < height; j++) {
				for (int j2 = 0; j2 < width; j2++) {
					cache.v[i*height*width+j*width+j2][0] = dout.v[i].v[j][j2];
				}
			}
		}
		return cache;
	}
	
	@Override
	public String toString() {
		return "UnFlatten()";
	}
}
