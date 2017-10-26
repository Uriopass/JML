package RL.environments;

import math.RandomGenerator;
import math.Vector;

public class GridWorld extends Environment {
	public int width, height;
	public int steps = 0;
	final int n_fruit = 2;
	final int n_walls = 2;
	
	public final int ACTION_UP    = 0;
	public final int ACTION_RIGHT = 1;
	public final int ACTION_DOWN  = 2;
	public final int ACTION_LEFT  = 3;
	
	static final int PLAYER_LAYER = 0;
	static final int WALL_LAYER   = 1;
	static final int TARGET_LAYER = 2;
	
	
	public int x_cur, y_cur;
	
	
	public GridWorld(int width, int height) {
		this.width = width;
		this.height = height;
		this.action_size = 4;
		this.state_size = this.width*this.height*3;
	}
	
	public double get_state_v(int f, int x, int y) {
		return state.v[f * this.width * this.height + y* this.width + x];
	}
	
	public void set_state_v(int f, int x, int y, double value) {
		state.v[f * this.width * this.height + y* this.width + x] = value;
	}
	
	
	
	@Override
	public void init() {
		state = new Vector(state_size);
		
		for(int i = 0 ; i < n_fruit ; i++) {
			set_state_v(TARGET_LAYER, RandomGenerator.uniform_int(width), RandomGenerator.uniform_int(height), 1);
		}
		for(int i = 0 ; i < n_walls ; i++) {
			int x, y;
			do {
				x = RandomGenerator.uniform_int(width);
				y = RandomGenerator.uniform_int(height);
			} while(get_state_v(TARGET_LAYER, x, y) == 1);
			set_state_v(WALL_LAYER, x, y, 1);
		}
		
		{
			do {
				x_cur = RandomGenerator.uniform_int(width);
				y_cur = RandomGenerator.uniform_int(height);
			} while(get_state_v(TARGET_LAYER, x_cur, y_cur) == 1 || get_state_v(WALL_LAYER, x_cur, y_cur) == 1);
			set_state_v(PLAYER_LAYER, x_cur, y_cur, 1);
		}
		steps = width*height;
		
		/*
		set_state_v(0, 1, 1, 1);
		set_state_v(2, RandomGenerator.uniform_int(0, 2)*2, RandomGenerator.uniform_int(0, 2)*2, 1);
		steps = 2;
		x_cur = 1;
		y_cur = 1;
		*/
		
	}
	
	public void print_state() {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int a = (int)get_state_v(PLAYER_LAYER, j, i);
				int b = (int)get_state_v(WALL_LAYER, j, i);
				int c = (int)get_state_v(TARGET_LAYER, j, i);
				
				if(a == 1)
					System.out.print("x");
				if(b == 1)
					System.out.print("-");
				if(c == 1)
					System.out.print("t");
				if(a+b+c == 0)
					System.out.print(".");
				System.out.print(" ");
			}
			System.out.println();
		}
	}

	public double move(int x, int y) {
		int next_x = x_cur + x;
		int next_y = y_cur + y;
		if(next_x >= 0 && next_x < width && next_y >= 0 && next_y < height) {
			if(get_state_v(WALL_LAYER, next_x, next_y) == 0) {
				set_state_v(PLAYER_LAYER, x_cur, y_cur, 0);
				set_state_v(PLAYER_LAYER, next_x, next_y, 1);
				x_cur = next_x;
				y_cur = next_y;
				double tmp = get_state_v(TARGET_LAYER, next_x, next_y);
				set_state_v(TARGET_LAYER, next_x, next_y, 0);
				return tmp+0.01;
			}
			return 0;
		}
		return -1;
	}
	
	@Override
	public double apply_action(int action) {
		steps -= 1;
		switch(action) {
		case ACTION_UP:
			return move(0, -1);
		case ACTION_DOWN:
			return move(0, 1);
		case ACTION_LEFT:
			return move(-1, 0);
		case ACTION_RIGHT:
			return move(1, 0);
		}
		return 0;
	}

	@Override
	public boolean is_terminal_state() {
		for(int i = 0 ; i < width ; i++)
			for(int j = 0 ; j < height ; j++)
				if(get_state_v(TARGET_LAYER, i, j) == 1)
					return steps <= 0;
		return true;
	}
}
