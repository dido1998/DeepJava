package basicops;
import java.util.*;

public class tensor
{
	public double data;
	public double grad;
	boolean trainable;
	public tensor(double d,boolean b)
	{
		data=d;
		grad=0;
		trainable=b;
	}
	
	public void applygrad(double learningrate)
	{
		if(trainable)
			data=data-learningrate*grad;
	}

}