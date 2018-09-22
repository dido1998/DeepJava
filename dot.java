package tensordef;
import java.math.*;
import basicops.*;
public class dot extends superopdef
{
	tensorarray weights;
	tensorarray inputs;
	tensorarray dotproduct;
	tensorarray weightssplit[];
	tensorarray inputssplit[];
	mul_tensorarray mulops[][];
	reduce_sum red_ops[][];
	tensorgraph graph;
	tensorarrayops op;
	tensorarray mulouts[][];
	backpropagationstructure<dot> curstruct;
	tensorarray redouts[][];
	public dot(tensorarray i1,tensorarray w,tensorgraph graph)
	{
		op=new tensorarrayops();
		weights=w;
		this.graph=graph;
		inputs=i1;
		dotproduct=new tensorarray(inputs.dim1,weights.dim2,false);
		//System.out.println(dotproduct);
		mulops=new mul_tensorarray[inputs.dim1][weights.dim2];
		red_ops=new reduce_sum[inputs.dim1][weights.dim2];
		mulouts=new tensorarray[inputs.dim1][weights.dim2];
		redouts=new tensorarray[inputs.dim1][weights.dim2];
		for(int i=0;i<inputs.dim1;i++)
		{
			for(int j=0;j<weights.dim2;j++)
			{
				mulouts[i][j]=new tensorarray(1,inputs.dim2,false);
				redouts[i][j]=new tensorarray(1,1,false);
			}
		}
		

		if (inputs.dim2!=weights.dim1 )
		{
			System.out.println("dimensions do not match");
			System.out.println("dimensions of parameter1:"+inputs.dim1 +" "+weights.dim2);
			System.out.println("dimensions of parameter2:"+inputs.dim1 +" "+weights.dim2);
			System.exit(1);
		}
		else
		{
			weightssplit=new tensorarray[weights.dim2];
			inputssplit=new tensorarray[inputs.dim1];
			op.split(weights,weightssplit,1,weights.trainable);
			op.split(inputs,inputssplit,0,inputs.trainable);
			for(int i=0;i<inputs.dim1;i++)
			{
				for(int j=0;j<weights.dim2;j++)
				{
					mulops[i][j]=new mul_tensorarray(inputssplit[i],weightssplit[j],graph);
					red_ops[i][j]=new reduce_sum(mulouts[i][j],graph);
				}
			}
			
		}
		curstruct=new backpropagationstructure<>(this,dotproduct,null);
		graph.addtolist(curstruct);
	}

	public tensorarray forward()
	{
		//System.out.println(this);
		for(int i=0;i<inputs.dim1;i++)
		{
			for(int j=0;j<weights.dim2;j++)
			{
				mulouts[i][j].assigntensorarray(mulops[i][j].forward());
				redouts[i][j].assigntensorarray(red_ops[i][j].forward());
			}
		}

		op.convertotensorarray(redouts,dotproduct,false);
		if(dotproduct.dim1!=inputs.dim1 && dotproduct.dim2!=weights.dim2)
		{
			System.out.println("error");
			System.exit(1);
		}
		
		//System.out.println(dotproduct);
		return dotproduct;

	}

	public void backward(tensorarray backflow)
	{
		//System.out.println(dotproduct.arr[0][0].grad);
		tensorarray opconverted[][]=new tensorarray[inputs.dim1][weights.dim2];
		//System.out.println(inputs.dim1+" "+weights.dim2);
		op.converttoarrayoftensorarray(backflow,opconverted,false);
		for(int i=0;i<inputs.dim1;i++)
		{
			for(int j=0;j<weights.dim2;j++)
			{
				//System.out.println(backflow.arr[i][j].grad);
				//System.out.println(opconverted[i][j]);
				red_ops[i][j].backward(opconverted[i][j]);
				
				///System.out.println(mulouts[i][j].arr[0][0].grad);
				mulops[i][j].backward(mulouts[i][j]);
			}
		}

		graph.removefromlist(curstruct);
	}
}