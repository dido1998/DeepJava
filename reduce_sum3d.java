package tensordef;
import basicops.*;
public class reduce_sum3d extends superopdef
{
	tensorgraph graph;
	backpropagationstructure<reduce_sum3d> curstruct;
	tensorarray3d arr;
	tensorarray3d eval[];
	add ops[];

	public reduce_sum3d(tensorarray3d arr,tensorgraph graph)
	{
		this.arr=arr;
		this.graph=graph;
		ops=new add[arr.dim1*arr.dim2*arr.dim3];
		eval=new tensorarray3d[arr.dim1*arr.dim2*arr.dim3+1];
		for(int i=0;i<=arr.dim1*arr.dim2*arr.dim3;i++)
		{
			eval[i]=new tensorarray3d(1,1,1,arr.trainable);
		}
		curstruct=new backpropagationstructure<>(this,null,eval[arr.dim1*arr.dim2*arr.dim3]);
		graph.addtolist(curstruct);
		int c=0;
		for(int i=0;i<arr.dim1;i++)
		{
			for(int j=0;j<arr.dim2;j++)
			{
				for(int k=0;k<arr.dim3;k++)
				{

					ops[c]=new add(arr.arr[i][j][k],eval[c].arr[0][0][0]);
					c++;
				}
			}
		}

	}

	public tensorarray3d forwardconv()
	{
		for(int i=0;i<arr.dim1*arr.dim2*arr.dim3;i++)
		{
			eval[i+1].arr[0][0][0].data=ops[i].forward().data;
		}
		return eval[arr.dim1*arr.dim2*arr.dim3];
	}

	public void backwardconv(tensorarray3d backflow)
	{
		//System.out.println(backflow.arr[0][0][0].grad);
		ops[arr.dim1*arr.dim2*arr.dim3-1].backward(backflow.arr[0][0][0]);

		for(int i=arr.dim1*arr.dim2*arr.dim3-2;i>=0;i--)
		{

			ops[i].backward(eval[i+1].arr[0][0][0]);
		}

		graph.removefromlist(curstruct);
	}

}