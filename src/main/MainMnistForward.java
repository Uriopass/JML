package main;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

import datareaders.MnistReader;
import image.ImageConverter;
import layers.Parameters;
import layers.activations.SigmoidActivation;
import layers.activations.SoftmaxActivation;
import layers.activations.TanhActivation;
import layers.flat.AffineLayer;
import layers.flat.BatchnormLayer;
import layers.flat.DenseLayer;
import layers.flat.SplitAffineLayer;
import layers.losses.EntropyLoss;
import layers.losses.SoftmaxCrossEntropy;
import math.Activations;
import math.Matrix;
import math.RandomGenerator;
import perceptron.MultiLayerPerceptron;

public class MainMnistForward {

	/* Les donnees */
	public static String path = "";
	public static String labelDB = path + "train-labels.idx1-ubyte";
	public static String imageDB = path + "train-images.idx3-ubyte";

	public static MultiLayerPerceptron model;

	// Nombre d'epoque max
	public final static int EPOCHMAX = 20;

	public static final int N_t = 20000;

	public static int T_t = 10000;

	public static int N;
	public static int T;

	public static Matrix trainData, testData;
	public static int[] trainRefs, testRefs;

	public static long seed = System.currentTimeMillis();

	public static final int[][] colors = { { 200, 32, 32 }, { 128, 128, 64 }, { 128, 110, 255 }, { 255, 255, 0 },
			{ 255, 0, 255 }, { 192, 255, 32 }, { 0, 0, 255 }, { 255, 255, 255 }, { 64, 128, 255 }, { 255, 128, 64 }, };

	public static void load_data() {
		N = N_t - (N_t % model.mini_batch);
		T = T_t - (T_t % model.mini_batch);

		System.out.println("# Loading the database !");
		/* Lecteur d'image */
		List<int[][]> images = MnistReader.getImages(imageDB);
		int[] refs = MnistReader.getLabels(labelDB);
		System.out.println("# Database loaded !");
		/* Taille des images et donc de l'espace de representation */
		int SIZEW = ImageConverter.image2VecteurReel(images.get(0)).length;

		/* Creation des donnees */
		trainData = new Matrix(SIZEW, N);
		trainRefs = new int[N];
		int cpt = 0;
		/* Donnees d'apprentissage */
		for (int l = 0; l < N; l++) {
			cpt++;
			trainData.v[l] = ImageConverter.image2VecteurReel_withB(images.get(l));
			int label = refs[l];
			trainRefs[l] = label;
		}

		System.out.println("# Train set " + cpt + " images");

		/* Donnees de test */
		System.out.println("# Build test");
		cpt = 0;
		final int TOTAL = images.size();
		if (N + T > TOTAL) {
			System.err.println("N+T (" + (N + T) + ") > Total (" + TOTAL + ")");
			throw new RuntimeException();
		}
		testData = new Matrix(SIZEW, T);
		testRefs = new int[T];
		for (int i = 0; i < T; i++) {
			cpt++;
			testData.v[i] = ImageConverter.image2VecteurReel_withB(images.get(N + i));
			int label = refs[N + i];
			testRefs[i] = label;
		}
		System.out.println("# Test set " + cpt + " images");
		trainData = trainData.transpose();
		testData = testData.transpose();
	}
	/*
	public static void visualize_bottleneck(String name) {
		int cellsize = 10;
		int imnum = T;
		int height = 1000;
		height = height - height % cellsize;
		int width = (int) (height * 16f / 9f);
		width = width - width % cellsize;
		int widthcell = width / cellsize;
		int heightcell = height / cellsize;
		int bottleneckdim = 3;
	
		ArrayList<Integer> ints = new ArrayList<Integer>();
		ArrayList<Vector> columns = new ArrayList<Vector>();
		int[] refs = new int[imnum];
		for (int i = 0; i < testData.width; i++) {
			ints.add(i);
			columns.add(testData.get_column(i));
		}
		Matrix batch = new Matrix(imnum, trainData.height);
		
		for (int j = 0; j < imnum; j++) {
			int indice = ints.get(j);
			batch.set_column(j, columns.get(indice));
			refs[j] = testRefs[indice];
		}
		int visu_layer = -1;
		for (int i = 0; i < model.layers.size() ; i++) {
			FlatLayer l = model.layers.get(i);
			if(l instanceof AffineLayer) {
				if (((AffineLayer)l).weight.height == bottleneckdim) {
					visu_layer = i + 2;
				}	
			}
		}
		if(visu_layer == -1)
			return;
		Matrix batch_cp = new Matrix(batch);
		for (int k = 0; k < visu_layer; k++) {
			batch = model.layers.get(k).forward(batch, false);
		}
		/*
		Matrix final_pos = new Matrix(batch);
		for (int k = visu_layer; k < model.layers.size(); k++) {
			final_pos = model.layers.get(k).forward(final_pos, false);
		}
		*/
	/*
	double max = 1.1;//batch.getRow(0).max();
	double min = -1.1;//batch.getRow(0).min();
	/*
	Matrix all = new Matrix(widthcell * heightcell, 2);
	for (int i = 0; i < heightcell; i++) {
		for (int j = 0; j < widthcell; j++) {
			all.v[0][i * widthcell + j] = minx + (maxx - minx) * ((float) (j) / widthcell);
			all.v[1][i * widthcell + j] = miny + (maxy - miny) * ((float) (i) / heightcell);
		}
	}
	//System.out.println(model.dims.length);
	for (int k = visu_layer; k < model.layers.size(); k++) {
		all = model.layers.get(k).forward(all, false);
	}
	*/
	/*
		BufferedImage bf;
		if(bottleneckdim == 2) {
			bf = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	
			Vector dec1 = new Vector(bottleneckdim);
			dec1.v[0] = -min;
			dec1.v[1] = -min;
			Vector scl = new Vector(bottleneckdim);
			scl.v[0] = (width) / (max - min);
			scl.v[1] = (height) / (max - min);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					//int indice = (i / cellsize) * widthcell + (j / cellsize);
					//int classe = all.get_column(indice).argmax();
					//double confidence = all.get_column(indice).v[classe];
					//System.out.println(classe);
					//int[] color = colors[classe];
					int r = 32;//(int) ((color[0] * confidence));
					int g = 32;//(int) ((color[1] * confidence));
					int b = 32;//(int) ((color[2] * confidence));
					int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;
	
					bf.setRGB(j, i, rgb);
				}
			}
			for (int i = 0; i < imnum; i++) {
				Vector init = batch_cp.get_column(i);
				Vector arrival = batch.get_column(i);
				//Vector theend = final_pos.get_column(i);
				int classe = testRefs[i];
				//double confidence = theend.v[classe];
				boolean isGoodClasse = classe == refs[i];
	
				arrival.add(dec1);
				arrival.scale(scl);
				for (int j = 0; j < 28; j++) {
					for (int k = 0; k < 28; k++) {
						int x = (int) (k - 14 + arrival.v[0]);
						int y = (int) (j - 14 + arrival.v[1]);
						if (x >= 0 && x < width && y >= 0 && y < height) {
							int indice = 28 * j + k;
							double color = init.v[indice];
	
							int oldcolor = bf.getRGB(x, y);
							float oldr = ((oldcolor >> 16) & 0xFF) / 255f;
							float oldg = ((oldcolor >> 8) & 0xFF) / 255f;
							float oldb = ((oldcolor >> 0) & 0xFF) / 255f;
	
							double newr = colors[classe][0]/255.0;//color;
							double newg = colors[classe][1]/255.0;//color;
							double newb = colors[classe][2]/255.0;//color;
							/*
							if (isGoodClasse) {
								newg = 1;
							} else {
								newr = 1;
							}
							*/
	/*
							double blend = color;//confidence * color;
							int r = (int) ((blend * newr + (1 - blend) * oldr) * 255);
							int g = (int) ((blend * newg + (1 - blend) * oldg) * 255);
							int b = (int) ((blend * newb + (1 - blend) * oldb) * 255);
							int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;
	
							bf.setRGB(x, y, rgb);
	
						}
					}
				}
			}
		}
		else if (bottleneckdim==3) {
			bf = new BufferedImage(width, height*3, BufferedImage.TYPE_INT_ARGB);
			
			Vector dec1 = new Vector(2);
			dec1.v[0] = -min;
			dec1.v[1] = -min;
			Vector scl = new Vector(2);
			scl.v[0] = (width) / (max - min);
			scl.v[1] = (height) / (max - min);
			for (int i = 0; i < height*3; i++) {
				for (int j = 0; j < width; j++) {
					//int indice = (i / cellsize) * widthcell + (j / cellsize);
					//int classe = all.get_column(indice).argmax();
					//double confidence = all.get_column(indice).v[classe];
					//System.out.println(classe);
					//int[] color = colors[classe];
					int r = 32;//(int) ((color[0] * confidence));
					int g = 32;//(int) ((color[1] * confidence));
					int b = 32;//(int) ((color[2] * confidence));
					int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;
	
					bf.setRGB(j, i, rgb);
				}
			}
			for (int i = 0; i < imnum; i++) {
				Vector init = batch_cp.get_column(i);
				Vector arrival = batch.get_column(i);
				//Vector theend = final_pos.get_column(i);
				int classe = testRefs[i];
				//double confidence = theend.v[classe];
				boolean isGoodClasse = classe == refs[i];
				for(int dim = 0 ; dim < 3 ; dim++) {
					for (int j = 0; j < 28; j++) {
						for (int k = 0; k < 28; k++) {
							Vector arr = new Vector(2);
							arr.v[0] = arrival.v[dim];
							arr.v[1] = arrival.v[(dim+1)%3];
							arr.add(dec1);
							arr.scale(scl);
							
							int x = (int) (k - 14 + arr.v[0]);
							int y = (int) (j - 14 + arr.v[1]);
							if (x >= 0 && x < width && y >= 0 && y < height) {
								int indice = 28 * j + k;
								double color = init.v[indice];
		
								int oldcolor = bf.getRGB(x, y);
								float oldr = ((oldcolor >> 16) & 0xFF) / 255f;
								float oldg = ((oldcolor >> 8) & 0xFF) / 255f;
								float oldb = ((oldcolor >> 0) & 0xFF) / 255f;
		
								double newr = colors[classe][0]/255.0;//color;
								double newg = colors[classe][1]/255.0;//color;
								double newb = colors[classe][2]/255.0;//color;
								/*
								if (isGoodClasse) {
									newg = 1;
								} else {
									newr = 1;
								}
								*/
	/*
							double blend = color;//confidence * color;
							int r = (int) ((blend * newr + (1 - blend) * oldr) * 255);
							int g = (int) ((blend * newg + (1 - blend) * oldg) * 255);
							int b = (int) ((blend * newb + (1 - blend) * oldb) * 255);
							int rgb = (0xFF << 24) + (g << 8) + (r << 16) + b;
	
							bf.setRGB(x, y+dim*height, rgb);
	
						}
					}
				}
			}
		}
	} else {
		return;
	}
	try {
		ImageIO.write(bf, "png", new File(name + ".png"));
	} catch (IOException e) {
		e.printStackTrace();
	}
	}
	*/

	public static void main(String[] args) {
		System.out.println("Appuyez sur ENTER pour d�marrer : ");
		long time = System.currentTimeMillis();
		RandomGenerator.init(seed);
		model = new MultiLayerPerceptron(64);
		load_data();
		Parameters p = new Parameters("reg=0.0001", "lr=0.005", "dout=false");
		model.add(new DenseLayer(784, 300, 0.3, "swish", true, p));
		p.set("dout", "true");
		model.add(new DenseLayer(300, 10, 0, "swish", false, p));
		model.add(new SoftmaxCrossEntropy());
		/*
		Parameters p = new Parameters("reg=0", "lr=0.001", "dout=false");
		model.add(new Unflatten(1, 28, 28));
		ConvolutionLayer cl = new ConvolutionLayer(28, 28, 1, 16, 5, 1, 0, p);
		p.set("dout", "true");
		MaxPooling mp = new MaxPooling(cl.width_out, cl.height_out, 3);
		model.add(cl);
		model.add(mp);
		model.add(new Flatten());
		model.add(new DenseLayer(mp.width_out * mp.height_out * cl.features_out, 128, 0, "tanh", false, p));
		model.add(new AffineLayer(128, 10, true, p));
		model.add(new SoftmaxCrossEntropy());
*/
		System.out.println("# Model created with following architecture : ");
		model.print_architecture();
	
		System.out.println("# Seed : " + seed);
		System.out.println("# Processors : " + Runtime.getRuntime().availableProcessors());

		double[] trainAccuracy = new double[EPOCHMAX + 1];
		double[] testAccuracy = new double[EPOCHMAX + 1];

		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
		otherSymbols.setDecimalSeparator('.');
		otherSymbols.setGroupingSeparator(',');
		DecimalFormat df = new DecimalFormat("#0.00", otherSymbols);

		//visualize_bottleneck("fig0");
		System.out.println("# Initialization took " + (System.currentTimeMillis() - time) + " ms");
		model.confusion_matrix(trainData, trainRefs).print_values();
		
		for (int i = 1; i <= EPOCHMAX; i++) {
			long t = System.currentTimeMillis();
			model.epoch(trainData, trainRefs);
			// model.writeDiff(testData, "auto"+i, 10);
			//visualize_bottleneck("fig"+i);
			double rms = (System.currentTimeMillis() - t) / 1000.;
			t = System.currentTimeMillis();
			testAccuracy[i] = (100. * model.correct_count(testData, testRefs)) / T;
			double test_forward_t = (System.currentTimeMillis() - t) / 1000.;
			trainAccuracy[i] = (100. * model.last_correct_count) / N;
			System.out.print(i + ((i >= 10) ? " " : "  "));
			System.out.print("Top 1 accuracy (train, test) : " + df.format(trainAccuracy[i]) + "% "
					+ df.format(testAccuracy[i]) + "% ");
			System.out.print("loss " + model.last_average_loss + "\t");
			System.out.print("epoch time " + df.format(rms) + "s");
			System.out.print("test time " + df.format(test_forward_t) + "s");
			System.out.println(" ETA " + df.format((EPOCHMAX - i) * (rms)) + "s");
			//model.confusion_matrix(trainData, trainRefs).print_values();
			
			System.out.println();
		}
		
		((AffineLayer)model.layers.get(0)).weight.visualize("test", 28, 1, 10, true);

		for (double f : trainAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
		for (double f : testAccuracy) {
			System.out.print(df.format(f) + ";");
		}
		System.out.println();
		// System.out.print("MLPerceptron On the test set : ");
		// System.out.println((100f * model.correct_count(testData, testRefs)) / T);
	}
}