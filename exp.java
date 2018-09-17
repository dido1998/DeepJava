package basicops;
import java.util.*;
import java.math.*;
public class exp
{
	tensor num1;
	
	public exp(tensor a)
	{
		num1=a;
		
	}
	public tensor forward()
	{
		tensor evaltensor=new tensor(Math.exp(num1.data),false);
		return evaltensor;
	}
	public void backward(tensor backflow)
	{
		num1.grad+=backflow.grad*Math.exp(num1.data);
	}
}