package basicops;

public class negate
{
	tensor data;
	public negate(tensor data)
	{
		this.data=data;
	}
	public tensor forward()
	{
		data.data=-data.data;
		return data;
	}
	public void backward(tensor backflow)
	{
		data.grad+=-backflow.grad;
	}
}