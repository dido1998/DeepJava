package tensordef;
import basicops.*;

public class mul_tensorarray3d extends superopdef
{
	tensorgraph graph;
	backpropagationstructure<mul_tensorarray3d> curstruct;
	tensorarray3d arr1;
	tensorarray3d arr2;
	tensorarray3d eval;
	mul mulops[][][];
	public mul_tensorarray3d(tensorarray3d arr1,tensorarray3d arr2,tensorgraph graph)
	{
		this.arr1=arr1;
		this.arr2=arr2;
		this.graph=graph;
		if(arr1.dim1!=arr2.dim1 && arr1.dim2!=arr2.dim2 && arr1.dim3!=arr2.dim3)
		{
			System.out.println("dimensions should be equal");
			System.exit(1);
		}
		else
		{
			eval=new tensorarray3d(arr1.dim1,arr1.dim2,arr1.dim3,arr1.trainable);
			curstruct=new backpropagationstructure<mul_tensorarray3d>(this,null,eval);
			graph.addtolist(curstruct);
			mulops=new mul[arr1.dim1][arr1.dim2][arr1.dim3];
			for(int i=0;i<arr1.dim1;i++)
			{
				for(int j=0;j<arr1.dim2;j++)
				{
					for(int k=0;k<arr1.dim3;k++)
					{

						mulops[i][j][k]=new mul(arr1.arr[i][j][k],arr2.arr[i][j][k]);
					}
				}
			}
			//System.out.println(arr1.arr[0][0][0].data);
			//System.out.println(arr2.arr[0][0][0].data);
		}
	}
	public tensorarray3d forwardconv()
	{
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				for(int k=0;k<arr1.dim3;k++)
				{
					eval.arr[i][j][k].data=mulops[i][j][k].forward().data;
				}
			}
		}
		return eval;
	}
	public void backwardconv(tensorarray3d backflow)
	{
		//System.out.println(backflow.arr[0][0][0].grad);
		for(int i=0;i<arr1.dim1;i++)
		{
			for(int j=0;j<arr1.dim2;j++)
			{
				for(int k=0;k<arr1.dim3;k++)
				{
					//System.out.println(arr1.arr);
					//System.out.println(arr2.arr);
					mulops[i][j][k].backward(backflow.arr[i][j][k]);
				}
			}
		}
			//System.out.println(arr1.arr[0][0][0].grad);
		graph.removefromlist(curstruct);	

	}
}