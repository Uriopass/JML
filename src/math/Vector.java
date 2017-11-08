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

	public Vector(double[] data) {
		v = data;
		length = v.length;
	}

	public Vector(int[] refs) {
		length = refs.length;
		v = new double[length];
		for(int i = 0 ; i < length ; i++) {
			v[i] = refs[i];
		}
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

	public Vector add(Vector b) {
		if (length != b.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + b.length + ")");
		}
		for (int i = 0; i < length; i++) {
			v[i] += b.v[i];
		}
		return this;
	}
	

	public Vector add(double val) {
		for (int i = 0; i < length; i++) {
			v[i] += val;
		}
		return this;
	}


	public static Vector power(Vector a, double pow) {
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

	public Vector power(double pow) {
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

	public Vector scale(Vector scalar) {
		if (length != scalar.length) {
			throw new RuntimeException("Incompatible shape : (" + length + ") and (" + scalar.length + ")");
		}
		for (int i = 0; i < length; i++) {
			v[i] *= scalar.v[i];
		}
		return this;
	}
	
	public Vector inverse() {
		for (int i = 0; i < length; i++) {
			v[i] = 1.0 / v[i];
		}
		return this;
	}

	public Vector scale(double scalar) {
		for (int i = 0; i < length; i++) {
			v[i] *= scalar;
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
	
	public Matrix to_column_matrix() {
		Matrix m = new Matrix(1, this.length);
		m.set_column(0, this);
		return m;
	}
	
	public Matrix to_row_matrix() {
		Matrix m = new Matrix(this.length, 1);
		m.set_row(0, this);
		return m;
	}
	
	public Vector append(Vector b) {
		   int aLen = length;
		   int bLen = b.length;
		   double[] c= new double[aLen+bLen];
		   System.arraycopy(v, 0, c, 0, aLen);
		   System.arraycopy(b.v, 0, c, aLen, bLen);
		   v = c;
		   return this;
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

	public void fill(double value) {
		for (int i = 0; i < v.length; i++) {
			v[i] = value;
		}
	}

	public double sum() {
		double sum = 0;
		for(int i = 0 ; i < v.length ; i++) {
			sum += v[i];
		}
		return sum;
	}

}
