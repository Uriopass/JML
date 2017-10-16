package image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.stream.ImageOutputStream;

import math.Matrix;

public class VisualizeWeightTime {
	public ArrayList<BufferedImage> bfs;
	public VisualizeWeightTime() {
		bfs = new ArrayList<>();
	}
	
	
	
	public void add(Matrix w, int dimension) {
		 bfs.add(w.visualize("", dimension, true, false));
	}
	
	public void AddVisualize(int layer) {
		
	}
	
	
	public void writeGif(String name) {
		try {
			File f = new File(name+".gif");
			f.createNewFile();
			ImageOutputStream ios = new FileImageOutputStream(f);
			GifSequenceWriter gsw = new GifSequenceWriter(ios, bfs.get(0).getType(), 100, true);
			for (BufferedImage bf : bfs) {
				gsw.writeToSequence(bf);
			}
			for (int i = 0; i < 10; i++) {
				gsw.writeToSequence(bfs.get(bfs.size() - 1));
			}
			gsw.close();
			ios.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
