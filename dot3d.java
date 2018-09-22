package tensordef;
import basicops.*;

public class dot3d extends superopdef
{
	tensorgraph graph;
	backpropagationstructure<dot3d> curstruct;
	tensorarray3d arr1;
	tensorarray3d arr2;
	tensorarray3d eval;
	tensorarray split1[];
	tensorarray split2[];
	tensorarrayops op;
	dot dotops[];
	tensorarray out[];
	public dot3d(tensorarray3d arr1,tensorarray3d arr2,tensorgraph graph)
	{
		op=new tensorarrayops();
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
			out=new tensorarray[arr1.dim3];
			//eval=new tensorarray3d(arr1.dim1,arr1.dim2,arr1.dim3,arr1.trainable);
			split1=op.convert3dto2d(arr1);
			split2=op.convert3dto2d(arr2);
			
			dotops=new dot[arr1.dim3];
			for(int i=0;i<arr1.dim3;i++)
			{
				dotops[i]=new dot(split1[i],split2[i],graph);
			}

			//System.out.println(arr1.arr[0][0][0].data);
			//System.out.println(arr2.arr[0][0][0].data);
		}
	}
	public tensorarray3d forwardconv()
	{
		for(int i=0;i<arr1.dim3;i++)
		{
			out[i]=dotops[i].forward();
		}
		eval=op.convert2dto3d(out);
		curstruct=new backpropagationstructure<dot3d>(this,null,eval);
		graph.addtolist(curstruct);
		return eval;
	}
	public void backwardconv(tensorarray3d backflow)
	{
		//System.out.println(backflow.arr[0][0][0].grad);
		for(int i=0;i<arr1.dim3;i++)
		{
			dotops[i].backward(out[i]);
			
		}
		graph.removefromlist(curstruct);	

	}
}