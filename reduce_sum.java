package tensordef;
import basicops.*;

public class reduce_sum extends superopdef
{
	tensorarray arr;
	tensorarray eval[];
	add addops[];
	backpropagationstructure<reduce_sum> curstruct;
	tensorgraph graph;
	public reduce_sum(tensorarray arr,tensorgraph graph)
	{
		this.arr=arr;
		this.graph=graph;
		eval=new tensorarray[arr.dim2+1];
		for(int i=0;i<arr.dim2+1;i++)
		{
		eval[i]=new tensorarray(1,1,false);
		}
		
		addops=new add[arr.dim2];
		for(int i=0;i<arr.dim2;i++)
		{
			addops[i]=new add(eval[i].arr[0][0],arr.arr[0][i]);
		}
		//System.out.println(eval[arr.dim2-1].arr[0][0]);
	}

	public tensorarray forward()
	{
		for(int i=0;i<arr.dim2;i++)
		{
			eval[i+1].arr[0][0].data=addops[i].forward().data;
		}
		//System.out.println(eval[0].arr);
		curstruct=new backpropagationstructure<>(this,eval[arr.dim2],null);
		graph.addtolist(curstruct);
		return eval[arr.dim2];
	}
	public void backward(tensorarray backflow)
	{
		//System.out.println(eval[arr.dim2].arr[0][0].grad);

		addops[arr.dim2-1].backward(backflow.arr[0][0]);
		//System.out.println(eval[arr.dim2-1].arr[0][0].grad);
		for(int i=arr.dim2-2;i>=0;i--)
		{
			//System.out.println(eval[i+1].arr[0][0].grad);
			addops[i].backward(eval[i+1].arr[0][0]);
		}
		graph.removefromlist(curstruct);
	}
}