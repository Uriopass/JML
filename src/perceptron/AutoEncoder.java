package perceptron;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import javax.imageio.ImageIO;

import layers.FlatLayer;
import layers.Layer;
import layers.flat.AffineLayer;
import layers.flat.BatchnormLayer;
import layers.losses.Loss;
import math.Matrix;
import math.RandomGenerator;
import math.Vector;
/**
 * Cette classe représente un autoencoder, qui apprends donc sans label à regénérer une image donnée, souvent utilisé pour réduire le nombre de dimensions utile.
 */
public class AutoEncoder extends FeedForwardNetwork {
	ArrayList<FlatLayer> layers;

	public AutoEncoder(int mini_batch) {
		this.mini_batch = mini_batch;
	}
	
	public void add(FlatLayer l) {
		layers.add(l);
	}

	@Override
	public Matrix forward(Matrix data) {
		Matrix next = new Matrix(data);
		for (FlatLayer l : layers) {
			next = l.forward(next, false);
		}
		return next;
	}

	/**
	 * Permet de visualiser un ensemble de donnée en affichant l'image d'origine et l'image reproduite.
	 * @param data données à visualiser
	 * @param name nom du fichier à écrire
	 * @param num nombre d'image à visualiser
	 */
	public void write_diff(Matrix data, String name, int num) {
		Matrix result = forward(data);
		int dimension = 28;
		int scale = 1;
		int height = num;
		BufferedImage bf = new BufferedImage(dimension * scale * 2, dimension * scale * height,
				BufferedImage.TYPE_INT_ARGB);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < dimension * scale; j++) {
				for (int k = 0; k < dimension * scale * 2; k++) {
					int indice = dimension * (j / scale) + (k % (dimension * scale)) / scale;
					int r, g, b;
					if (k >= dimension * scale) {
						r = (int) (255 * data.v[indice][i]);
						g = (int) (255 * data.v[indice][i]);
						b = (int) (255 * data.v[indice][i]);
					} else {
						r = (int) (255 * result.v[indice][i]);
						g = (int) (255 * result.v[indice][i]);
						b = (int) (255 * result.v[indice][i]);
					}

					int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;
					bf.setRGB(k, i * dimension * scale + j, rgb);
				}
			}
		}
		try {
			ImageIO.write(bf, "png", new File(name + ".png"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Effectue une epoch à partir d'un ensemble de donnée
	 * @param data
	 */
	public void epoch(Matrix data) {
		// On mélange les données
		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();

		for (int i = 0; i < data.width; i++) {
			ints.add(i);
			columns.add(data.get_column(i));
		}
		Collections.shuffle(ints, RandomGenerator.r);

		System.out.print("[");
		int tenth = (data.width / mini_batch) / 10;

		// On itère sur les mini_batch
		for (int i = 0; i < data.width / mini_batch; i++) {
			Matrix batch = new Matrix(mini_batch, data.height);
			if (i % tenth == 0)
				System.out.print("=");

			// On génère le mini_batch
			for (int j = 0; j < mini_batch; j++) {
				int indice = ints.get(i * mini_batch + j);
				batch.set_column(j, columns.get(indice));
			}

			Matrix batch_init = new Matrix(batch);
			
			// Propagation avant
			for (int j = 0; j < layers.size(); j++) {
				batch = layers.get(j).forward(batch, true);
			}

			// On donne l'erreur
			Matrix dout = batch;
			Loss l = get_loss_layer();
			l.feed_ref(batch_init);

			// Propagation arrière
			for (int j = layers.size() - 1; j >= 0; j--) {
				dout = layers.get(j).backward(dout);
				layers.get(j).apply_gradient();
			}
			last_average_loss += l.loss;
		}
		last_average_loss /= data.width / mini_batch;

		// Fin d'époque
		for (FlatLayer l : layers) {
			if (l instanceof BatchnormLayer) {
				((BatchnormLayer) l).end_of_epoch();
			}
			if (l instanceof AffineLayer) {
				((AffineLayer) l).end_of_epoch();
			}
		}
		System.out.print("] ");
	}

	@Override
	public Loss get_loss_layer() {
		return (Loss) layers.get(layers.size() - 1);
	}

	@Override
	public void print_architecture() {
		for (Layer l : layers) {
			System.out.println("# - " + l);
		}
	}
}
