package perceptron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import math.Activations;
import math.Matrix;
import math.Vector;

public class Perceptron {
	public static int DIM; // dimension de l'espace de représentation
	public static Matrix w; // paramètres du modèle
	public static Matrix data;
	public static int num_classes;
	public static int[] refs; // les réferences
	public static float learning_rate = 0.1f;
	public static float reg = 0.003f;
	// Calcule le resultat du perceptron
	public static Vector forward(Vector datapoint) {
		return w.dot(datapoint);
	}

	public static void xavier(long seed) {
		Random r = new Random(seed);
		w = new Matrix(DIM, num_classes);
		double mult = 1 / Math.sqrt(DIM);
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
		Vector res = Activations.softmax(forward(datapoint));
		res.v[true_ref] -= 1;
		for (int j = 0; j < num_classes; j++) {
			for (int i = 0; i < DIM; i++) {
				grad.v[j][i] = datapoint.v[i] * res.v[j] + w.v[j][i] * reg;
			}
		}

		return grad;
	}

	// Calcule la correction à appliquer au poids en fonction d'un point
	public static Matrix w_grad_vectorized(Matrix datapoints, int[] true_refs) {
		//System.out.println("Datapoint shape : "+datapoints.shape());
		//System.out.println("w shape : "+w.shape());
		Matrix res = Activations.softmax(w.parralel_mult_transposed(datapoints), 0);
		//System.out.println("res shape"+res.shape());
		for(int i = 0 ; i < true_refs.length ; i++) {
			res.v[i][true_refs[i]] -= 1;
		}
		
		Matrix grad = datapoints.parralel_mult_transposed(res);
		grad.addInPlace(w.scale(reg*datapoints.width)); // L2 loss
		/*
		for (int j = 0; j < num_classes; j++) {
			for (int i = 0; i < DIM; i++) {
				grad.v[j][i] = datapoint.v[i] * (res.v[j] - one_hot.v[j]) + w.v[j][i] * reg;
			}
		}
		*/

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
		Matrix res = Activations.softmax(w.parralel_mult(data), 1);
		for (int i = 0; i < data.width; i++) {
			Vector r = res.get_column(i);
			if (argmax(r) == refs[i]) {
				counter++;
			}
		}
		return counter;
	}

	// Met à jour les poids en passant par toutes les données une fois
	public static void epoch() {
		int mini_batch = 1;
		ArrayList<Integer> ints = new ArrayList<Integer>();
		for(int i = 0 ; i < data.width ; i++) {
			ints.add(i);
		}
		Collections.shuffle(ints);
		for (int i = 0; i < data.width/mini_batch; i++) { 
			Matrix grads = new Matrix(w.width, w.height);
			for(int j = 0 ; j < mini_batch ; j++) {
				int indice = ints.get(i*mini_batch+j);
				Vector datapoint = data.get_column(indice);
				int ref = refs[indice];
				grads.addInPlace(w_grad(datapoint, ref));
			}
			grads.scaleInPlace(-learning_rate/mini_batch);
			w.addInPlace(grads);
		}
	}
	public static void epoch_vectorized() { 
		int mini_batch = 16;
		ArrayList<Integer> ints = new ArrayList<Integer>();
		for(int i = 0 ; i < data.width ; i++) {
			ints.add(i);
		}
		Collections.shuffle(ints);
		for (int i = 0; i < data.width/mini_batch; i++) {
			Matrix batch = new Matrix(mini_batch, w.width);
			int[] refs_v = new int[mini_batch];
			for(int j = 0 ; j < mini_batch ; j++) {
				int indice = ints.get(i*mini_batch+j);
				batch.set_column(j, data.get_column(indice));
				refs_v[j] = refs[indice];
			}
			Matrix grads = w_grad_vectorized(batch, refs_v);
			grads.scaleInPlace(-learning_rate/mini_batch);
			w.addInPlace(grads);
		}
	}
}