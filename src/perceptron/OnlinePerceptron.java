package perceptron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import math.Matrix;
import math.Vector;

public class OnlinePerceptron {
	public static int DIM; // dimension de l'espace de représentation
	public static Matrix w; // paramètres du modèle
	public static Matrix data;
	public static int num_classes;
	public static int[] refs; // les réferences
	public static float learning_rate = 0.1f;
	public static float reg = 0.0003f;

	// In-place
	public static Vector sigmoid(Vector x) {
		for (int i = 0; i < x.length; i++) {
			x.v[i] = (float) (1 / (1 + Math.exp(-x.v[i])));	
		}
		return x;
	}

	// Calcule le resultat du perceptron
	public static Vector forward(Vector datapoint) {
		return w.dot(datapoint);
	}

	public static Vector softmax(Vector activations) {
		double max = activations.max();
		
		float expSum = 0;
		for (int i = 0; i < activations.length; i++) {
			activations.v[i] = (float) Math.exp(activations.v[i]-max);
			expSum += activations.v[i];
		}
		for (int i = 0; i < activations.length; i++) {
			activations.v[i] /= expSum;
		}
		return activations;
	}

	public static void xavier() {
		Random r = new Random();
		w = new Matrix(DIM, num_classes);
		double mult = 1 / Math.sqrt(DIM * num_classes);
		System.out.println("mult:" + mult);
		for (int i = 0; i < w.height; i++) {
			for (int j = 0; j < w.width; j++) {
				w.v[i][j] = r.nextGaussian() * mult;
			}
		}
	}

	// Calcule la correction à appliquer au poids en fonction d'un point
	public static Matrix w_grad(Vector datapoint, int true_ref) {
		Matrix grad = new Matrix(DIM, num_classes);
		Vector res = softmax(forward(datapoint));
		Vector one_hot = new Vector(num_classes);
		one_hot.v[true_ref] = 1;
		for (int j = 0; j < num_classes; j++) {
			for (int i = 0; i < DIM; i++) {
				grad.v[j][i] = datapoint.v[i] * (res.v[j] - one_hot.v[j]) + w.v[j][i] * reg;
			}
		}

		return grad;
	}

	public static int argmax(Vector v) {
		int max = 0;
		double maxV = v.v[0];
		for (int i = 1; i < v.length; i++) {
			if (v.v[i] > maxV) {
				maxV = v.v[i];
				max = i;
			}
		}
		return max;
	}

	// Compte le nombre de points classifiés correctement
	public static int correct_count() {
		int counter = 0;
		for (int i = 0; i < data.height; i++) {
			Vector res = softmax(forward(data.getRow(i)));
			if (argmax(res) == refs[i])
				counter++;
		}
		return counter;
	}

	// Met à jour les poids en passant par toutes les données une fois
	public static void epoch() {
		int mini_batch = 10;
		ArrayList<Integer> ints = new ArrayList<Integer>();
		for(int i = 0 ; i < data.height ; i++) {
			ints.add(i);
		}
		Collections.shuffle(ints);
		for (int i = 0; i < data.height/mini_batch; i++) {
			Matrix grads = new Matrix(w.width, w.height);
			for(int j = 0 ; j < mini_batch ; j++) {
				int indice = ints.get(i*mini_batch+j);
				Vector datapoint = data.getRow(indice);
				int ref = refs[indice];
				grads.addInPlace(w_grad(datapoint, ref));
			}
			grads.scaleInPlace(-learning_rate/mini_batch);
			w.addInPlace(grads);
		}
	}
}