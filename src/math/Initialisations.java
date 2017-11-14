package math;

/**
 * Liste d'initilisations pour des matrices et vecteur, pour l'instant la liste est courte et ne contient que xavier et he_uniform
 */
public class Initialisations {
	// he_uniform prend un nombre au hasard uniformément dans l'intervalle [-var ; var] avec var = sqrt(6 / fan_in)
	public static void he_uniform(Matrix m, int fan_in, double multiplier) {
		double bound = multiplier * Math.sqrt(6f / (fan_in));

		for (int i = 0; i < m.height; i++) {
			for (int j = 0; j < m.width; j++) {
				m.v[i][j] = RandomGenerator.uniform(-bound, bound);
			}
		}
	}

	// xavier prend un nombre au hasard selon une distribution gaussienne avec comme variance var = sqrt(2 / (fan_in + fan_out))
	public static void xavier(Matrix m, int fan_in, int fan_out) {
		double var = Math.sqrt(2f / (fan_in + fan_out));
		for (int i = 0; i < m.height; i++) {
			for (int j = 0; j < m.width; j++) {
				m.v[i][j] = RandomGenerator.gaussian(var);
			}
		}
	}

	public static void he_uniform(Vector v, int fan_in, double multiplier) {
		double bound = multiplier * Math.sqrt(6f / (fan_in));

		for (int j = 0; j < v.length; j++) {
			v.v[j] = RandomGenerator.uniform(-bound, bound);
		}
	}
}
