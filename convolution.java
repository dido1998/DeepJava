package tensordef;
import basicops.*;

public class convolution extends superopdef
{
	tensorarray3d arr;
	tensorarray3d filters[];
	tensorgraph graph;
	tensorarray3d paddedinputs;
	tensorarray3d eval;
	int numfilters;
	int padding;
	int filtersize;
	tensorarray3d back[][][];
	tensorarrayops ops;
	tensorarray3d eval1[][][];
	backpropagationstructure<convolution> curstruct; 
	mul_tensorarray3d mulops[][][];
	reduce_sum3d redops[][][];
	tensorarray3d ipslice[][][];
	public convolution(tensorarray3d arr,int filtersize,int numfilters,tensorgraph graph,String pad)
	{
		padding=0;
		ops=new tensorarrayops();
		this.arr=arr;
		this.numfilters=numfilters;
		this.graph=graph;
		this.filtersize=filtersize;
		if(pad.equals("SAME"))
		{
			padding=(filtersize-1)/2;
			paddedinputs=ops.pad(arr,arr.dim1+2*padding,arr.dim2+2*padding);
		}else
		{
			paddedinputs=arr;
		}



		eval=new tensorarray3d((arr.dim1-filtersize+2*padding)+1,arr.dim2-filtersize+2*padding+1,numfilters,false);
		back=new tensorarray3d[(arr.dim1-filtersize+2*padding)+1][arr.dim2-filtersize+2*padding+1][numfilters];
		eval1=new tensorarray3d[numfilters][(arr.dim1-filtersize+2*padding)+1][arr.dim2-filtersize+2*padding+1];
		ipslice=new tensorarray3d[numfilters][eval.dim1][eval.dim2];
		filters=new tensorarray3d[numfilters];
		for(int i=0;i<numfilters;i++)
		{
			filters[i]=new tensorarray3d(filtersize,filtersize,arr.dim3,true);
		}
		mulops=new mul_tensorarray3d[numfilters][eval.dim1][eval.dim2];
		redops=new reduce_sum3d[numfilters][eval.dim1][eval.dim2];
		for(int k=0;k<numfilters;k++)
		{
			for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
			{
				for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
				{
					ipslice[k][i][j]=ops.getslices(paddedinputs,i,i+filtersize,j,j+filtersize);
					back[i][j][k]=new tensorarray3d(1,1,1,eval.trainable);
					eval1[k][i][j]=new tensorarray3d(filtersize,filtersize,paddedinputs.dim3,paddedinputs.trainable);
					mulops[k][i][j]=new mul_tensorarray3d(ipslice[k][i][j],filters[k],graph);
					redops[k][i][j]=new reduce_sum3d(eval1[k][i][j],graph);
				}
			}
		}
		curstruct=new backpropagationstructure<convolution>(this,null,eval);
		graph.addtolist(curstruct);
	}

	public tensorarray3d forwardconv()
	{
		for(int k=0;k<numfilters;k++)
		{
			for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
			{
				for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
				{
					eval1[k][i][j].assigntensorarray(mulops[k][i][j].forwardconv());
					eval.arr[i][j][k].data=redops[k][i][j].forwardconv().arr[0][0][0].data;
				}
			}
		}
		return eval;
	}

	public void backwardconv(tensorarray3d backflow)
	{
		//System.out.println(backflow.arr[0][0][0].grad);
		ops.converttoarraytenorarray3d(backflow,back);
		//System.out.println(back[0][0][0].arr[0][0][0].grad);
		for(int k=0;k<numfilters;k++)
		{
			for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
			{
				for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
				{
					redops[k][i][j].backwardconv(back[i][j][k]);
					//System.out.println("hello");
					mulops[k][i][j].backwardconv(eval1[k][i][j]);
					
				}
			}
		}
		//System.out.println(ipslice[0][0][0].arr[0][0][0]);
		//System.out.println(paddedinputs.arr[0][0][0].grad);
		graph.removefromlist(curstruct);
	}	
}