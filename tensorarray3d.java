package tensordef;
import basicops.*;
import java.util.Random;

public class tensorarray3d
{
	public tensor arr[][][];
	public int dim1;
	public int dim2;
	public int dim3;
	boolean trainable;
	public tensorarray3d(int dim1,int dim2,int dim3,boolean trainable)
	{
		this.dim1=dim1;
		this.dim2=dim2;
		this.dim3=dim3;
		this.trainable=trainable;
		arr=new tensor[dim1][dim2][dim3];
		randominitialize();
	} 
	public void randominitialize()
	{
		Random rand = new Random();
		
		for (int i=0;i<dim1;i++)
		{
			for (int j=0;j<dim2;j++)
			{
				for(int k=0;k<dim3;k++)
				{
				arr[i][j][k]=new tensor(rand.nextDouble(),trainable);
				//System.out.println(arr[i][j].data);
				}
			}
		}
	}
	public void ones()
	{	
		for (int i=0;i<dim1;i++)
		{
			for (int j=0;j<dim2;j++)
			{
				for(int k=0;k<dim3;k++)
					arr[i][j][k].data=1;
				//System.out.println(arr[i][j].data);
			}
		}		
	}
	public void zeros()
	{
		for(int i=0;i<dim1;i++)
		{
			for(int j=0;j<dim2;j++)
			{
				for(int k=0;k<dim3;k++)
				{
					arr[i][j][k].data=0;
				}
			}
		}
	}
	public void assign(double customdata[][][])
	{
		for(int i=0;i<dim1;i++)
		{
			for(int j=0;j<dim2;j++)
			{
				for(int k=0;k<dim3;k++)
					arr[i][j][k].data=customdata[i][j][k];
			}
		}
	}
	public void assigntensorarray(tensorarray3d t)
	{
		for(int i=0;i<dim1;i++)
		{
			for(int j=0;j<dim2;j++)
			{
				for(int k=0;k<dim3;k++)

					arr[i][j][k].data=t.arr[i][j][k].data;
			}
		}
	}
	public void assigntensor(tensor t,int i,int j,int k)
	{
		arr[i][j][k]=null;
		arr[i][j][k]=t;
	}
}