package basicops;
import java.util.*;
import java.math.*;
public class div
{
	tensor num1;
	tensor num2;
	public div(tensor a,tensor b)
	{
		num1=a;
		num2=b;
	}
	public tensor forward()
	{
		tensor evaltensor=new tensor(num1.data/num2.data,false);
		return evaltensor;
	}
	public void backward(tensor backflow)
	{
		num1.grad+=backflow.grad*(1/num2.data);
		num2.grad+=backflow.grad*(-num1.data/Math.pow(num2.data,2));
	}
}