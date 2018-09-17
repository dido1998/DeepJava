package tensordef;
import java.util.*;
import java.util.Random;
import basicops.*;

import java.math.*;

public class tensorarray
{
	public tensor arr[][];
	
	public int dim1, dim2;
	public boolean trainable;
	public tensorarray(int dim1,int dim2,boolean trainable)
	{
		arr=new tensor[dim1][dim2];
		this.dim1=dim1;
		this.dim2=dim2;
		this.trainable=trainable;
		randomintialize();
	}
	public void randomintialize()
	{
		Random rand = new Random();
		for (int i=0;i<dim1;i++)
		{
			for (int j=0;j<dim2;j++)
			{
				arr[i][j]=new tensor(rand.nextDouble(),trainable);
				//System.out.println(arr[i][j].data);
			}
		}
	}
	
	public void ones()
	{	
		for (int i=0;i<dim1;i++)
		{
			for (int j=0;j<dim2;j++)
			{
				arr[i][j].data=1;
				//System.out.println(arr[i][j].data);
			}
		}		
	}
	public void assign(double customdata[][])
	{
		for(int i=0;i<dim1;i++)
		{
			for(int j=0;j<dim2;j++)
			{
				arr[i][j].data=customdata[i][j];
			}
		}
	}
	public void assigntensorarray(tensorarray t)
	{
		for(int i=0;i<dim1;i++)
		{
			for(int j=0;j<dim2;j++)
			{
				arr[i][j].data=t.arr[i][j].data;
			}
		}
	}
	public void assign(tensor ip)
	{
		arr[0][0]=ip;

	}
}
