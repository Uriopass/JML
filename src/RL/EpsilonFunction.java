package RL;

public interface EpsilonFunction {
	public abstract double epsilon(int t, int total);
	
	public static EpsilonFunction linear_decrease = new EpsilonFunction(){
		@Override
		public double epsilon(int t, int total) {
			return 1-(double)(t)/total;
		}
	};
	
	public static EpsilonFunction square_decrease = new EpsilonFunction(){
		@Override
		public double epsilon(int t, int total) {
			return (1-(double)(t)/total)*(1-(double)(t)/total);
		}
	};
	
	public static EpsilonFunction constant = new EpsilonFunction() {
		@Override
		public double epsilon(int t, int total) {
			return 1;
		}
	};
}
