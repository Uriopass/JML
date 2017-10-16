package math;

public class Vector {
	public int length;
	public double[] v;

	public Vector(Vector datapoint) {
		v = new double[datapoint.length];
		this.length = datapoint.length;
		for(int i = 0 ; i < length ; i++)
			v[i] = datapoint.v[i];
	}
	
	public Vector(int length) {
		v = new double[length];
		this.length = length;
	}

	public Vector dot(Vector b) {
		return dot(this, b);
	}

	public static Vector dot(Vector a, Vector b) {
		if (a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length + ") and (" + b.length + ")");
		}
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] * b.v[i];
		}
		return res;
	}

	public Vector add(Vector b) {
		return add(this, b);
	}

	public static Vector add(Vector a, Vector b) {
		if (a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length + ") and (" + b.length + ")");
		}
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] + b.v[i];
		}
		return res;
	}

	public static Matrix outer(Vector a, Vector b) {
		Matrix m = new Matrix(b.length, a.length);
		for(int i = 0 ; i < m.height ; i++) {
			for(int j = 0 ; j < m.width ; j++) {
				m.v[i][j] = a.v[i]*b.v[j];
			}
		}
		return m;
	}
	
	public Vector scale(Vector scalar) {
		return Vector.scale(this, scalar);
	}

	public Vector scale(double scalar) {
		return Vector.scale(this, scalar);
	}

	public static Vector scale(Vector a, Vector b) {
		if (a.length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + a.length + ") and (" + b.length + ")");
		}
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] * b.v[i];
		}
		return res;
	}

	public static Vector scale(Vector a, double scalar) {
		Vector res = new Vector(a.length);
		for (int i = 0; i < a.length; i++) {
			res.v[i] = a.v[i] * scalar;
		}
		return res;
	}

	public Vector addInPlace(Vector b) {
		if (length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + b.length + ")");
		}
		for (int i = 0; i < length; i++) {
			v[i] += b.v[i];
		}
		return this;
	}
	

	public Vector addInPlace(double val) {
		for (int i = 0; i < length; i++) {
			v[i] += val;
		}
		return this;
	}

	public Vector scaleInPlace(double scalar) {
		for (int i = 0; i < length; i++) {
			v[i] *= scalar;
		}
		return this;
	}

	public Vector power(double pow) {
		return Vector.power(this, pow);
	}

	private static Vector power(Vector a, double pow) {
		Vector res = new Vector(a.length);
		if (pow == 0.5) {
			for (int i = 0; i < a.length; i++) {
				res.v[i] = Math.sqrt(a.v[i]);
			}
		} else {
			for (int i = 0; i < a.length; i++) {
				res.v[i] = Math.pow(a.v[i], pow);
			}

		}
		return res;
	}

	public Vector powerInPlace(double pow) {
		if (pow == 0.5) {
			for (int i = 0; i < length; i++) {
				v[i] = Math.sqrt(v[i]);
			}
		} else {
			for (int i = 0; i < length; i++) {
				v[i] = Math.pow(v[i], pow);
			}
		}
		return this;
	}

	public Vector scaleInPlace(Vector scalar) {
		if (length != scalar.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + scalar.length + ")");
		}
		for (int i = 0; i < length; i++) {
			v[i] *= scalar.v[i];
		}
		return this;
	}

	public int argmax() {
		int max = 0;
		double maxV = v[0];
		for (int i = 1; i < v.length; i++) {
			if (v[i] > maxV) {
				maxV = v[i];
				max = i;
			}
		}
		return max;
	}
	
	public double max() {
		double max = v[0];
		for (int i = 1; i < v.length; i++)
			max = Math.max(max, v[i]);
		return max;
	}

	public double min() {
		double min = v[0];
		for (int i = 1; i < v.length; i++)
			min = Math.min(min, v[i]);
		return min;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append('(');
		for (int i = 0; i < this.length - 1; i++) {
			sb.append(this.v[i]);
			sb.append(", ");
		}
		sb.append(this.v[this.length - 1]);
		sb.append(')');
		return sb.toString();
	}

}
