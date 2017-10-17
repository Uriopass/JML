package layers;

import java.util.HashMap;

public class Parameters {
	public HashMap<String, String> values;
	public Parameters(String...strings) {
		values = new HashMap<String, String>();
		for(String s : strings) {
			String[] keyval = s.split("=");
			if(keyval.length != 2) {
				throw new RuntimeException("Error parsing "+s);
			}
			values.put(keyval[0], keyval[1]);
		}
	}
	
	public String get(String key) {
		return values.get(key);
	}
	
	public String set(String key, String value) {
		return values.put(key, value);
	}
	
	public String getAsString(String key, String default_value) {
		if(values.containsKey(key))
			return values.get(key);
		return default_value;
	}
	
	public double getAsDouble(String key, double default_value) {
		if(values.containsKey(key))
			return Double.parseDouble(values.get(key));
		return default_value;
	}
}
