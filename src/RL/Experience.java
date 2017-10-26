package RL;

import math.Vector;

public class Experience {
	public Vector s, next_s;
	public double r;
	public int a;
	public boolean is_terminal;
}
