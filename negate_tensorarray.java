package tensordef;
import basicops.*;
import java.util.*;

public class negate_tensorarray extends superopdef
{
	tensorarray arr;
	tensorarray eval;
	backpropagationstructure<negate_tensorarray> curstruct;
	negate ops[][];
	tensorgraph graph;
	public negate_tensorarray(tensorarray arr,tensorgraph graph)
	{
		this.arr=arr;
		this.graph=graph;
		eval=new tensorarray(arr.dim1,arr.dim2,arr.trainable);
		curstruct=new backpropagationstructure<>(this,eval,null);
		graph.addtolist(curstruct);
		ops=new negate[arr.dim1][arr.dim2];
		for(int i=0;i<arr.dim1;i++)
		{
			for(int j=0;j<arr.dim2;j++)
			{
				ops[i][j]=new negate(arr.arr[i][j]);
			}
		}
	}
	public tensorarray forward()
	{
		for(int i=0;i<arr.dim1;i++)
		{
			for(int j=0;j<arr.dim2;j++)
			{
				eval.arr[i][j].data=ops[i][j].forward().data;
			}
		}
		return eval;
	}
	public void backward(tensorarray backward)
	{
		for(int i=0;i<arr.dim1;i++)
		{
			for(int j=0;j<arr.dim2;j++)
			{
				ops[i][j].backward(backward.arr[i][j]);
			}
		}
		graph.removefromlist(curstruct);
	}
}