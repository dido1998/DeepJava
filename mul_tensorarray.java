package tensordef;
import basicops.*;
import java.math.*;
import java.util.*;
public class mul_tensorarray extends superopdef
{
	tensorarray arr1;
	tensorarray arr2;
	mul mulops[][];
	tensorarray eval;
	tensorgraph graph;
	backpropagationstructure<mul_tensorarray> curstruct;
	public mul_tensorarray(tensorarray t1,tensorarray t2,tensorgraph graph)
	{
		arr1=t1;
		arr2=t2;
		this.graph=graph;
		eval=new tensorarray(arr1.dim1,arr1.dim2,false);
		curstruct=new backpropagationstructure<mul_tensorarray>(this,eval,null);
		graph.addtolist(curstruct);
		if (arr1.dim1!=arr2.dim1 && arr1.dim2!=arr2.dim2)
		{
			System.out.println("dimensions do not match");
			System.out.println("dimensions of parameter1:"+arr1.dim1 +" "+arr1.dim2);
			System.out.println("dimensions of parameter2:"+arr2.dim1 +" "+arr2.dim2);
			System.exit(1);
		}
		else
		{
		mulops=new mul[arr1.dim1][arr1.dim2];
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				mulops[i][j]=new mul(arr1.arr[i][j],arr2.arr[i][j]);
			}
		}
		}
	}
	public tensorarray forward()
	{
		
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				eval.arr[i][j].data=mulops[i][j].forward().data;	
			}
		}
		return eval;		
	}
	public void backward(tensorarray backflow)
	{
		//System.out.println(backflow.arr[0][0].grad);
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				mulops[i][j].backward(backflow.arr[i][j]);
			}
		}

		graph.removefromlist(curstruct);
	}
}