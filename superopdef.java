package tensordef; 
import java.util.*;

public class superopdef
{

	public tensorarray forward()
	{
		System.out.println("wrong datatype\n");
		System.exit(1);
		return null;
	}
	public tensorarray3d forwardconv()
	{
		System.out.println("wrong datatype\n");
		System.exit(1);
		return null;
	}
	public void backward(tensorarray backflow)
	{
		System.out.println("not implemented\n");
		System.exit(1);
	}
	public void backwardconv(tensorarray3d backflow)
	{
		System.out.println("not implemented\n");
		System.exit(1);
	}}