package basicops;
import java.util.*;
import java.math.*;
public class log
{
	tensor num1;
	
	public log(tensor a)
	{
		num1=a;
		
	}
	public tensor forward()
	{
		tensor evaltensor=new tensor(Math.log(num1.data),false);
		return evaltensor;
	}
	public void backward(tensor backflow)
	{
		num1.grad+=backflow.grad*(1/num1.data);
	}
}