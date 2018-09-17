package tensordef;
import basicops.*;
import java.math.*;
import java.util.*;

public class exp_tensorarray extends superopdef
{
	tensorarray arr1;
	exp expops[][];
	tensorarray eval;
	tensorgraph graph;
	backpropagationstructure<exp_tensorarray> curstruct;
	public exp_tensorarray(tensorarray t1,tensorgraph graph)
	{
		arr1=t1;	
		this.graph=graph;
		eval=new tensorarray(arr1.dim1,arr1.dim2,false);	
		expops=new exp[arr1.dim1][arr1.dim2];
		curstruct=new backpropagationstructure<exp_tensorarray>(this,eval,null);
		graph.addtolist(curstruct);
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				expops[i][j]=new exp(arr1.arr[i][j]);				
			}
		}
	}
	public tensorarray forward()
	{
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				eval.arr[i][j].data=expops[i][j].forward().data;	
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
				expops[i][j].backward(backflow.arr[i][j]);
			}
		}
		graph.removefromlist(curstruct);
	}
}
