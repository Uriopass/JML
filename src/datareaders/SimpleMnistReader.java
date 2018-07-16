package datareaders;

import java.io.File;
import java.util.List;

import image.ImageConverter;
import math.Matrix;
import math.Vector;


public class SimpleMnistReader {
	private static boolean loaded = false;
	
	// change theses before calling get*() to change the validation ration
	public static int N = 30016;
	public static int V = 9984;
	
	public static final int T = 10000;
	
	static final int out_size = 10;

	public static String train_labelDB = "train-labels.idx1-ubyte";
	public static String train_imageDB = "train-images.idx3-ubyte";

	public static String test_labelDB = "t10k-labels.idx1-ubyte";
	public static String test_imageDB = "t10k-images.idx3-ubyte";
	
	public static Matrix train_data, validation_data, test_data;
	
	public static Matrix train_refs, validation_refs, test_refs;
	
	private static void testPath(String path) {
		System.out.println(new File(path).getAbsolutePath());
		if (!new File(path).exists())
			throw new RuntimeException(path + " not found, please check the path.");
	}
	
	public static void init() {
		if(loaded)
			return;
		
		System.out.println("[LOAD] Loading MNIST database");
		long time = System.currentTimeMillis();
		testPath(train_imageDB);
		testPath(test_imageDB);
		
		testPath(train_labelDB);
		testPath(test_labelDB);
		
		List<int[][]> train_images = MnistReader.getImages(train_imageDB);
		int[] all_train_refs = MnistReader.getLabels(train_labelDB);

		List<int[][]> test_images = MnistReader.getImages(test_imageDB);
		int[] all_test_refs = MnistReader.getLabels(test_labelDB);
		System.out.println("[LOAD] MNIST database loaded in " + (System.currentTimeMillis() - time) + "ms, converting to matrices");

		int SIZEW = 28 * 28;

		/* Creation des donnees */
		train_data = new Matrix(N, SIZEW);
		train_refs = new Matrix(N, out_size);

		validation_data = new Matrix(V, SIZEW);
		validation_refs = new Matrix(V, out_size);

		if (N + V > 40000) {
			throw new RuntimeException("[ERROR][LOAD] N+V (" + (N + V) + ") > 40'000, too much data asked");
		}

		for (int l = 0; l < N + V; l++) {
			int label = all_train_refs[l];
			double[] image = ImageConverter.image2VecteurReel(train_images.get(l));
			if (l < N) {
				train_data.set_column(l, new Vector(image));
				train_refs.set_column(l, Vector.one_hot(out_size, label));
			} else {
				validation_data.set_column(l-N, new Vector(image));
				validation_refs.set_column(l-N, Vector.one_hot(out_size, label));
			}
		}

		System.out.println("[LOAD] Train/Validation set for MNIST built with " + N + " train and " + V + " validation images");

		test_data = new Matrix(T, SIZEW);
		test_refs = new Matrix(T, out_size);
		for (int i = 0; i < T; i++) {
			test_data.set_column(i, new Vector(ImageConverter.image2VecteurReel(test_images.get(i))));
			int label = all_test_refs[i];
			test_refs.set_column(i, Vector.one_hot(out_size, label));
		}
		System.out.println("[LOAD] Test set for MNIST built with " + T + " images");
		
		loaded = true;
	}
	
	public static Matrix getTrainData() {
		init();
		return train_data;
	}
	
	public static Matrix getValidationData() {
		init();
		return validation_data;
	}
	
	public static Matrix getTestData() {
		init();
		return test_data;
		
	}
	
	public static Matrix getTrainRefs() {
		init();
		return train_refs;
	}
	
	public static Matrix getValidationRefs() {
		init();
		return validation_refs;
	}
	
	public static Matrix getTestRefs() {
		init();
		return test_refs;
	}
}