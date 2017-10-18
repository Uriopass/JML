package math;

public class Initialisations {
	public static void he_uniform(Matrix m, int fan_in, double multiplier) {
		double bound = multiplier * Math.sqrt(6f / (fan_in));
		
		for (int i = 0; i < m.height; i++) {
			for (int j = 0; j < m.width; j++) {
				m.v[i][j] = RandomGenerator.uniform(-bound, bound);
			}
		}
	}
	
	public static void xavier(Matrix m, int fan_in, int fan_out) {
		double var = Math.sqrt(2 / (fan_in + fan_out));
		for (int i = 0; i < m.height; i++) {
			for (int j = 0; j < m.width; j++) {
				m.v[i][j] = RandomGenerator.gaussian(var);
			}
		}
	}
}
