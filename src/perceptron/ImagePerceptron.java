package perceptron;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import javax.imageio.ImageIO;

import image.ImageConverter;
import math.Activations;
import math.Matrix;
import math.Vector;
import mnist.MnistReader;

public class ImagePerceptron {

	/* Les donnees */
	public static String path = "";
	public static String labelDB = path + "t10k-labels.idx1-ubyte";
	public static String imageDB = path + "t10k-images.idx3-ubyte";

	public static MultiLayerPerceptron model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 7;

	public static final int N_t = 0;//50000;

	public static int T_t = 10000-1;

	public static int N;
	public static int T;

	public static Matrix trainData, testData;
	public static int[] trainRefs, testRefs;
	public static int SIZEW = 0;

	public static long seed = System.currentTimeMillis();

	public static final int[][] colors = { { 200, 32, 32 }, { 128, 128, 64 }, { 128, 110, 255 }, { 255, 255, 0 },
			{ 255, 0, 255 }, { 192, 255, 32 }, { 0, 0, 255 }, { 255, 255, 255 }, { 64, 128, 255 }, { 255, 128, 64 }, };

	public static void load_mnist_data() {
		N = N_t;// - (N_t % model.mini_batch);
		T = T_t;// - (T_t % model.mini_batch);

		System.out.println("# Loading the database !");
		/* Lecteur d'image */
		List<int[][]> images = MnistReader.getImages(imageDB);
		int[] refs = MnistReader.getLabels(labelDB);
		System.out.println("# Database loaded !");
		/* Taille des images et donc de l'espace de representation */
		SIZEW = ImageConverter.image2VecteurReel(images.get(0)).length;

		/* Creation des donnees */
		trainData = new Matrix(SIZEW + 1, N);
		trainRefs = new int[N];
		int cpt = 0;
		/* Donnees d'apprentissage */
		for (int l = 1; l <= N; l++) {
			cpt++;
			trainData.v[l - 1] = ImageConverter.image2VecteurReel_withB(images.get(l));
			int label = refs[l];
			trainRefs[l - 1] = label;
		}

		System.out.println("# Train set " + cpt + " images");

		/* Donnees de test */
		System.out.println("# Build test");
		cpt = 0;
		final int TOTAL = images.size();
		if (N + T >= TOTAL) {
			System.err.println("N+T ("+(N+T)+") > Total ("+TOTAL+")");
			throw new RuntimeException();
		}
		testData = new Matrix(SIZEW + 1, T);
		testRefs = new int[T];
		for (int i = 1; i <= T; i++) {
			cpt++;
			testData.v[i - 1] = ImageConverter.image2VecteurReel_withB(images.get(N + i));
			int label = refs[N + i];
			testRefs[i - 1] = label;
		}
		System.out.println("# Test set " + cpt + " images");
		trainData = trainData.transpose();
		testData = testData.transpose();
	}

	public static void train_perceptron() {
		Perceptron.DIM = SIZEW + 1;
		Perceptron.num_classes = 10;
		Perceptron.data = trainData;
		Perceptron.refs = trainRefs;
		Perceptron.xavier(seed);

		for (int i = 0; i < 50; i++) {
			Perceptron.epoch_vectorized();
			System.out.println(i);
		}
		
		System.out.print("Perceptron On the test set : ");
		Perceptron.data = testData;
		Perceptron.refs = testRefs;

		System.out.println((100f * Perceptron.correct_count()) / T);
	}
	
	public static void visualize_bottleneck(String name) {
		int cellsize = 10;
		int imnum = T;
		int height = 1000;
		height = height - height % cellsize;
		int width = (int) (height * 16f / 9f);
		width = width - width % cellsize;
		int widthcell = width / cellsize;
		int heightcell = height / cellsize;

		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();
		int[] refs = new int[imnum];
		for (int i = 0; i < testData.width; i++) {
			ints.add(i);
			columns.add(testData.get_column(i));
		}
		Matrix batch = new Matrix(imnum, model.dims[0]);
		for (int j = 0; j < imnum; j++) {
			int indice = ints.get(j);
			batch.set_column(j, columns.get(indice));
			refs[j] = testRefs[indice];
		}
		int visu_layer = -1;
		for (int i = 0; i < model.weights.length; i++) {
			if (model.weights[i].height == 2) {
				visu_layer = i + 1;
			}
		}

		Matrix batch_cp = new Matrix(batch);
		for (int k = 0; k < visu_layer; k++) {
			batch = model.forward_layer(batch, k);
			model.activation_forward(batch);
		}
		Matrix final_pos = new Matrix(batch);
		for (int k = visu_layer; k < model.dims.length - 2; k++) {
			final_pos = model.forward_layer(final_pos, k);
			model.activation_forward(final_pos);
		}
		final_pos = model.forward_layer(final_pos, model.dims.length - 2);
		Activations.softmax(final_pos, 1);
		double maxx = 1.1;//batch.getRow(0).max();
		double minx = -1.1;//batch.getRow(0).min();
		double maxy = 1.1;//batch.getRow(1).max();
		double miny = -1.1;//batch.getRow(1).min();

		Matrix all = new Matrix(widthcell * heightcell, 2);
		for (int i = 0; i < heightcell; i++) {
			for (int j = 0; j < widthcell; j++) {
				all.v[0][i * widthcell + j] = minx + (maxx - minx) * ((float) (j) / widthcell);
				all.v[1][i * widthcell + j] = miny + (maxy - miny) * ((float) (i) / heightcell);
			}
		}
		//System.out.println(model.dims.length);
		for (int k = visu_layer; k < model.dims.length - 2; k++) {
			all = model.forward_layer(all, k);
			model.activation_forward(all);
		}
		all = model.forward_layer(all, model.dims.length - 2);
		Activations.softmax(all, 1);

		BufferedImage bf = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

		Vector dec1 = new Vector(2);
		dec1.v[0] = -minx;
		dec1.v[1] = -miny;
		Vector scl = new Vector(2);
		scl.v[0] = (width) / (maxx - minx);
		scl.v[1] = (height) / (maxy - miny);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int indice = (i / cellsize) * widthcell + (j / cellsize);
				int classe = model.argmax(all.get_column(indice));
				double confidence = all.get_column(indice).v[classe];
				//System.out.println(classe);
				int[] color = colors[classe];
				int r = (int) ((color[0] * confidence));
				int g = (int) ((color[1] * confidence));
				int b = (int) ((color[2] * confidence));
				int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;

				bf.setRGB(j, i, rgb);
			}
		}

		for (int i = 0; i < imnum; i++) {
			Vector init = batch_cp.get_column(i);
			Vector arrival = batch.get_column(i);
			Vector theend = final_pos.get_column(i);
			int classe = model.argmax(theend);
			double confidence = theend.v[classe];
			boolean isGoodClasse = classe == refs[i];

			arrival.addInPlace(dec1);
			arrival.scaleInPlace(scl);
			for (int j = 0; j < 28; j++) {
				for (int k = 0; k < 28; k++) {
					int x = (int) (k - 14 + arrival.v[0]);
					int y = (int) (j - 14 + arrival.v[1]);
					if (x >= 0 && x < width && y >= 0 && y < height) {
						int indice = 1 + 28 * j + k;
						double color = init.v[indice];

						int oldcolor = bf.getRGB(x, y);
						float oldr = ((oldcolor >> 16) & 0xFF) / 255f;
						float oldg = ((oldcolor >> 8) & 0xFF) / 255f;
						float oldb = ((oldcolor >> 0) & 0xFF) / 255f;

						double newr = 0;//color;
						double newg = 0;//color;
						double newb = 0;//color;
						if (isGoodClasse) {
							newg = 1;
						} else {
							newr = 1;
						}

						double blend = confidence * color;
						int r = (int) ((blend * newr + (1 - blend) * oldr) * 255);
						int g = (int) ((blend * newg + (1 - blend) * oldg) * 255);
						int b = (int) ((blend * newb + (1 - blend) * oldb) * 255);
						int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;

						bf.setRGB(x, y, rgb);

					}
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
	 * @param args
	 */
	public static void main(String[] args) {
		long time = System.currentTimeMillis();
		model = new MultiLayerPerceptron();
		
		load_mnist_data();
		System.out.println("# Seed : "+seed);
		System.out.println("# Processors : "+Runtime.getRuntime().availableProcessors());

		model.data = trainData;
		model.refs = trainRefs;
		model.dims = new int[] { SIZEW + 1, 800, 10 };

		double[] trainAccuracy = new double[EPOCHMAX + 1];
		double[] testAccuracy = new double[EPOCHMAX + 1];

		//model.weight_init(seed);
		model.load_weights("005percent");
		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(','); 
		DecimalFormat df = new DecimalFormat("#0.00", otherSymbols);
		

		System.out.println("# Initialization took "+(System.currentTimeMillis()-time)+" ms");
		/*
		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			model.epoch_rmsprop();
			double rms = (System.currentTimeMillis()-t)/1000.;
			t = System.currentTimeMillis();
			testAccuracy[i] = 100-(100. * model.correct_count(testData, testRefs)) / T;
			double test_forward_t = (System.currentTimeMillis()-t)/1000.;
			trainAccuracy[i] = 100-(100. * model.last_correct_count) / N;
			System.out.print(i+"\tTop 1 error rates (train, test) : "+df.format(trainAccuracy[i])+"% "+df.format(testAccuracy[i])+"%\t");
			System.out.print("epoch time "+df.format(rms)+"s test time "+df.format(test_forward_t)+"s");
			System.out.println(" ETA "+df.format((EPOCHMAX-i)*(test_forward_t+rms))+"s");
			model.write_weights("temp");
		}

		for (double f : trainAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
		for (double f : testAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
	*/
		System.out.print("MLPerceptron On the test set : ");
		System.out.println((100f * model.correct_count(testData, testRefs)) / T);
	}
}