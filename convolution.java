package tensordef;
import basicops.*;

public class convolution extends superopdef
{
	tensorarray3d arr;
	tensorarray filters[];
	tensorgraph graph;
	tensorarray3d paddedinputs;
	tensorarray3d eval;
	int numfilters;
	int padding;
	int filtersize;
	tensorarray back[][][];
	tensorarrayops ops;
	tensorarray3d eval1[][][];
	backpropagationstructure<convolution> curstruct; 
	dot3d mulops[][][];
	dot dotops[][][];
	reduce_sum3d redops[][][];
	tensorarray ipslice[][];
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

		//System.out.println(filtersize);
		eval=new tensorarray3d((arr.dim1-filtersize+2*padding)+1,arr.dim2-filtersize+2*padding+1,numfilters,false);
		back=new tensorarray[(arr.dim1-filtersize+2*padding)+1][arr.dim2-filtersize+2*padding+1][numfilters];
		ipslice=new tensorarray[eval.dim1][eval.dim2];
		filters=new tensorarray[numfilters];

		//
		
		for(int i=0;i<numfilters;i++)
		{
			filters[i]=new tensorarray(1,filtersize*filtersize*arr.dim3,true);
		}
		//filters[0].print();
		dotops=new dot[eval.dim1][eval.dim2][numfilters];
		for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
		{
			for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
			{
				ipslice[i][j]=ops.stretch(ops.getslices(paddedinputs,i,i+filtersize,j,j+filtersize),true);
				for(int k=0;k<numfilters;k++)
				{
					//[k].print();
					//System.out.println(k);

					back[i][j][k]=new tensorarray(1,1,eval.trainable);
					//System.out.println(back[i][j][k]);
					dotops[i][j][k]=new dot(filters[k],ipslice[i][j],graph);
				}
			}
		}


		curstruct=new backpropagationstructure<convolution>(this,null,eval);
		graph.addtolist(curstruct);
		
	}

	public tensorarray3d forwardconv()
	{
		for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
		{
			for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
			{
				for(int k=0;k<numfilters;k++)
				{
					eval.arr[i][j][k].data=dotops[i][j][k].forward().arr[0][0].data;
				}
			}
		}
		// System.out.println("hello");
		return eval;
	}

	public void backwardconv(tensorarray3d backflow)
	{
		//System.out.println("bcfgsg");
		//backflow.print();
		ops.tensorarray3dtoarrayoftensorarray2d(backflow,back);

		for(int i=0;i<=paddedinputs.dim1-filtersize;i++)
		{
			for(int j=0;j<=paddedinputs.dim2-filtersize;j++)
			{
				for(int k=0;k<numfilters;k++)
				{
					//System.out.println(back[i][j][k]);
					//System.out.println("----------------");	
					dotops[i][j][k].backward(back[i][j][k]);
				}
			}
		}		
		graph.removefromlist(curstruct);
	}	
}