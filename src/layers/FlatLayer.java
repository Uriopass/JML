package layers;

import math.Matrix;

/**
 * Interface g�n�rique de couche plate.
 */
public interface FlatLayer extends Layer {
	/**
	 * @param in Donn�es d'entr�e de la forme (d, n). d = dimension d'entr�e, n = nombre de donn�es
	 * @param training Indique si l'on va entra�ner cette couche ou non
	 * @return out Donn�es de sortie apr�s passage par la couche
	 */
	public abstract Matrix forward(Matrix in, boolean training);

	/**
	 * @param dout D�riv�e partiel du r�sultat de la couche
	 * @return D�riv�e partiel des donn�es d'entr�e de la couche
	 */
	public abstract Matrix backward(Matrix dout, boolean train);
}
