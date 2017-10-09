package math;

public class Vector {
	public int length;
	public double[] v;
	
	public Vector(int length) {
		v = new double[length];
		this.length = length;
	}
	
	public Vector dot(Vector b) {
		return dot(this, b);
	}
	
	public static Vector dot(Vector a, Vector b) {
		if(a.length != b.length) {
			throw new RuntimeException("Incompatible shape : ("+a.length+") and ("+b.length+")");
		}
		Vector res = new Vector(a.length);
		for(int i = 0 ; i < a.length ; i++) {
			res.v[i] = a.v[i]*b.v[i];
		}
		return res;
	}
	
	public Vector add(Vector b) {
		return add(this, b);
	}
	
	public static Vector add(Vector a, Vector b) {
		if(a.length != b.length) {
			throw new RuntimeException("Incompatible shape : ("+a.length+") and ("+b.length+")");
		}
		Vector res = new Vector(a.length);
		for(int i = 0 ; i < a.length ; i++) {
			res.v[i] = a.v[i]+b.v[i];
		}
		return res;
	}
	
	public Vector scale(double scalar) {
		return Vector.scale(this, scalar);
	}
	
	public static Vector scale(Vector a, double scalar) {
		Vector res = new Vector(a.length);
		for(int i = 0 ; i < a.length ; i++) {
			res.v[i] = a.v[i]*scalar;
		}
		return res;
	}
	
	public Vector addInPlace(Vector b) {
		if(length != b.length) {
			throw new RuntimeException("Incompatible shape : ("+length+") and ("+b.length+")");
		}
		for(int i = 0 ; i < length ; i++) {
			v[i] += b.v[i];
		}
		return this;
	}
	
	public Vector scaleInPlace(double scalar) {
		for(int i = 0 ; i < length ; i++) {
			v[i] *= scalar;
		}
		return this;
	}

	public double max() {
		double max = v[0];
		for(int i = 1 ; i < v.length ; i++)
			max = Math.max(max, v[i]);
		return max;
	}
}
