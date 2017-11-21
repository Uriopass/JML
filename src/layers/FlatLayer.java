package layers;

import math.Matrix;

/**
 * Interface générique de couche plate.
 */
public interface FlatLayer extends Layer {
	/**
	 * @param in Données d'entrée de la forme (d, n). d = dimension d'entrée, n = nombre de données
	 * @param training Indique si l'on va entraîner cette couche ou non
	 * @return out Données de sortie après passage par la couche
	 */
	public abstract Matrix forward(Matrix in, boolean training);

	/**
	 * @param dout Dérivée partiel du résultat de la couche
	 * @return Dérivée partiel des données d'entrée de la couche
	 */
	public abstract Matrix backward(Matrix dout, boolean train);
}
