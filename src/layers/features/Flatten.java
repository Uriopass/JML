package layers.features;

import layers.Layer;
import math.FeatureMatrix;
import math.Matrix;

public class Flatten implements Layer {
	public int features, width, height, outlength;
	FeatureMatrix cache;
	public Matrix forward(FeatureMatrix in, boolean training) {
		width = in.width;
		height = in.height;
		features = in.features;
		cache = in;
		Matrix result = new Matrix(1, features*height*width);
		for (int i = 0; i < features; i++) {
			for (int j = 0; j < height; j++) {
				for (int j2 = 0; j2 < width; j2++) {
					result.v[i*height*width+j*width+j2][0] = in.v[i].v[j][j2];
				}
			}
		}
		return result;
	}
	
	public static int getLength(int features, int height, int width) {
		return width*height*features;
	}

	public FeatureMatrix backward(Matrix dout) {
		for (int i = 0; i < features; i++) {
			for (int j = 0; j < height; j++) {
				for (int j2 = 0; j2 < width; j2++) {
					cache.v[i].v[j][j2] = dout.v[i*height*width+j*width+j2][0];
				}
			}
		}
		return cache;
	}
	
	@Override
	public String toString() {
		return "Flatten()";
	}
}
