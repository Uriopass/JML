package layers.reccurent;

import java.util.HashMap;

import math.Matrix;
import math.Vector;

public class RNNCellCache {
	public HashMap<String, Matrix> matrices;
	public HashMap<String, Vector> vectors;
	
	public RNNCellCache() {
		matrices = new HashMap<>();
		vectors = new HashMap<>();
	}
	
	public void remember(String name, Matrix m) {
		this.matrices.put(name, m);
	}
	
	public void remember(String name, Vector v) {
		this.vectors.put(name, v);
	}
	
	public Matrix get_m(String name) {
		return matrices.get(name);
	}
	
	public Vector get_v(String name) {
		return vectors.get(name);
	}
}
