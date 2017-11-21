package RL;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Random;

import math.Vector;

public class BanditTest {
	public int T; // Horizon
	public int n; // Nombre de bras
	public double[] Q; // Estimation de la fonction de valeur Q
	public double[] paramB; // parametre des bras
	public int[] nbTireBras; // Nb de fois que j'ai tiré un bras

	Random choix;

	// Constructeur : specifie le nombre de bras et l'horizon!
	public BanditTest(int T_, int n_) {
		T = T_;
		n = n_;
		Q = new double[n];
		paramB = new double[n];
		nbTireBras = new int[n];
	}

	// Renvoie la récompense lorsqu'on tire le bras b
	public int TireBras(int b) {
		nbTireBras[b] += 1;
		if (choix.nextDouble() < paramB[b])
			return 1;
		return 0;
	}

	// Initialise les différentes variables de la classe bandit
	public void Init(long seed) {
		// Init VA
		choix = new Random(seed);
		// Init Q
		for (int i = 0; i < Q.length; i++) {
			Q[i] = 0;
		}
		// Init paramB
		for (int i = 0; i < paramB.length; i++) {
			paramB[i] = choix.nextDouble();
		}
		// Init nbTireBras
		for (int i = 0; i < nbTireBras.length; i++) {
			nbTireBras[i] = 0;
		}
	}

	// Retourne l'indice du maximum d'une liste
	public int argmax(double[] list) {
		return new Vector(list).argmax();
	}

	/*
	 * Implementation de epsilon-greedy - choix du bras - renvoie la recompense
	 */
	public double ChoixBras_EpsG(double epsilon) {
		// TODO
		int bras;
		if (choix.nextDouble() < epsilon) {
			bras = choix.nextInt(n);
		} else {
			bras = argmax(Q);
		}
		double reward = TireBras(bras);

		MAJ_Q(bras, reward, 1);

		return reward;
	}

	/*
	 * Mise à jour de la fonction Q - alpha : parametre d'apprentissage
	 */
	public void MAJ_Q(int bras, double rew, double alpha) {
		Q[bras] = (Q[bras] * (nbTireBras[bras] - 1) + rew) / nbTireBras[bras];
	}

	/*
	 * Effectue une expérience (on tire T bras avec parametre epsilon) - renvoie
	 * le gain total en fonction de t
	 */
	public double[] Experiment(double eps) {
		// init gain vs t
		double[] gain = new double[T];
		for (int i = 0; i < T; i++)
			gain[i] = 0;

		gain[0] = ChoixBras_EpsG(eps);
		for (int i = 1; i < T; i++) {
			gain[i] = gain[i - 1] + ChoixBras_EpsG(eps);
		}

		return gain;
	}

	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

		BanditTest bt;

		int Horizon = 1000;
		int n_bras = 10;
		Vector av_gain = new Vector(Horizon);

		bt = new BanditTest(Horizon, n_bras);

		// Calcul le gain total moyen en fonction du temps
		for (int xp = 0; xp < 1000; xp++) {
			// TODO : on initialise les bras et appelle Experiment
			bt.Init(xp);
			av_gain.add(new Vector(bt.Experiment(0.1f)));
		}

		// Ecrit le resultat dans un fichier
		PrintWriter f = new PrintWriter("MesBellesDonnes.d", "UTF-8");
		for (int i = 0; i < av_gain.length; i++)
			f.println(i + " " + av_gain.v[i] / Horizon);
		f.close();
	}
}
