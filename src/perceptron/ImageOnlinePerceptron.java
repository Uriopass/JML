package perceptron;
 
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import image.ImageConverter;
import math.Matrix;
import math.Vector;
import mnisttools.MnistReader;
 
public class ImageOnlinePerceptron {
 
	/* Les donnees */
	public static String path="";
	public static String labelDB=path+"train-labels-idx1-ubyte";
	public static String imageDB=path+"train-images-idx3-ubyte";
 
	/* Parametres */
	// les N premiers exemples pour l'apprentissage
	public static final int N = 20000; 
	// les T derniers exemples  pour l'evaluation
	public static final int T = 3000; 
	// Nombre d'epoque max
	public final static int EPOCHMAX=25;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.err.println("# Load the database !");
		/* Lecteur d'image */ 
		MnistReader db = new MnistReader(labelDB, imageDB);
		/* Taille des images et donc de l'espace de representation */
		final int SIZEW = ImageConverter.image2VecteurReel(db.getImage(1)).length;
 
 
		/* Creation des donnees */
		System.err.println("# Build training");
		Matrix trainData = new Matrix(SIZEW+1, N);
		int[] trainRefs = new int[N];
		int cpt=0;
		/* Donnees d'apprentissage */
		for (int i = 1; i <= N; i++) {
			cpt++;
			trainData.v[i-1]=ImageConverter.image2VecteurReel_withB(db.getImage(i));
			int label = db.getLabel(i);
			trainRefs[i-1] = label;
		}
		System.err.println("Train set with "+cpt);
 
		/* Donnees de test */
		System.err.println("# Build test");
		cpt=0;
		final int TOTAL = db.getTotalImages();
		if (N+T >= TOTAL){
			System.out.println("N+T > Total");
			throw new RuntimeException();
		}
		Matrix testData = new Matrix(SIZEW+1, T);
		int[] testRefs = new int[T];
		for (int i = 0; i < T; i++) {
			cpt++;
			testData.v[i]=ImageConverter.image2VecteurReel_withB(db.getImage(TOTAL-i));
			int label = db.getLabel(TOTAL-i);
			testRefs[i]=label;
		}
		System.err.println("Test set with "+cpt);
 
		OnlinePerceptron.DIM = SIZEW+1;
		OnlinePerceptron.num_classes = 10;
		OnlinePerceptron.data = trainData;
		OnlinePerceptron.refs = trainRefs;
		OnlinePerceptron.xavier();
		int[] valid = new int[EPOCHMAX];
		valid[0] = OnlinePerceptron.correct_count();
		for(int i = 1 ; i < EPOCHMAX ; i++) {
			OnlinePerceptron.epoch();
			valid[i] = OnlinePerceptron.correct_count();
			System.out.println(valid[i]);
		}
		System.out.println("At initialisation: "+((100f*valid[0])/N));
		for(int i = 0 ; i < EPOCHMAX-1 ; i++) {
			if(valid[i+1] > valid[i]) {
				System.out.println("At epoch "+(i+1)+": "+((100f*valid[i+1])/N));
			}
		}
		System.out.print("On the test set : ");
		OnlinePerceptron.data = testData;
		OnlinePerceptron.refs = testRefs;
		System.out.println((100f*OnlinePerceptron.correct_count())/T);
		
		
		for(int i = 0 ; i < 10 ; i++) {
			Matrix w = OnlinePerceptron.w;
			double max = w.max();
			double min = w.min();

			BufferedImage bf;
			bf = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
			for(int j = 0 ; j < 28 ; j++) {
				for(int k = 0 ; k < 28 ; k++) {
					int indice = 28*j+k+1;
					
					int v = (int) (512*(w.v[i][indice]-min)/(max-min));
					v -= 256;
					int a=0, b=0;
					if(v < 0)
						a = -v;
					else
						b = v;
					
					a *= 1.5;
					b *= 1.5;
					if(a > 255)
						a = 255;
					if(b > 255)
						b = 255;
					int rgb = (0xFF << 24) + (0 << 8) + (a << 16)+b;
					bf.setRGB(k, j, rgb);
					//System.out.print(v);
					
					/*
					Vector ds = new Vector(3);
					ds.v[1] = -Math.abs(w.v[i][indice]);
					ds.v[0] = -Math.abs(w.v[i][indice]-min);
					ds.v[2] = -Math.abs(w.v[i][indice]-max);
					int d = OnlinePerceptron.argmax(ds);
					if(d == 0)
						System.out.print(".");
					if(d==1)
						System.out.print("-");
					if(d==2)
						System.out.print("^");
				*/
				}
				//System.out.println();
			}
			try {
				ImageIO.write(bf, "png", new File("test"+i+".png"));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println("********************");
		}
		
	}
 
 
}