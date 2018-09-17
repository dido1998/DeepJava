package tensordef;
import basicops.*;
import java.util.*;
import java.math.*;

public class tensorarrayops
{
	public void split(tensorarray t,tensorarray result[],int dim,boolean trainable)
	{
		
		int size=result.length;
		for(int i=0;i<size;i++)
		{
			if(dim==0)
			{
				tensorarray temp=new tensorarray(1,t.dim2,trainable);
				for(int j=0;j<t.dim2;j++)
				{
					temp.arr[0][j]=t.arr[i][j];
				}
				result[i]=temp;
			}else
			{
				tensorarray temp=new tensorarray(1,t.dim1,trainable);
				for(int j=0;j<t.dim1;j++)
				{
					temp.arr[0][j]=t.arr[j][i];
				}
				result[i]=temp;

			}
		}



	}
	public tensorarray convertotensorarray(tensorarray ip[][],boolean trainable)
	{
		tensorarray op=new tensorarray(ip[0].length,ip.length,trainable);
		for(int i=0;i<ip[0].length;i++)
		{
			for(int j=0;j<ip.length;j++)
			{
				op.arr[i][j]=ip[i][j].arr[0][0];
			}
		}
		return op;
	}

	public void converttoarrayoftensorarray(tensorarray ip,tensorarray op[][],boolean trainable)
	{

		for(int i=0;i<ip.dim1;i++)
		{
			for(int j=0;j<ip.dim2;j++)
			{
				op[i][j]=new tensorarray(1,1,trainable);
				op[i][j].assign(ip.arr[i][j]);
			}
		}
	}
	public void converttoarraytenorarray3d(tensorarray3d ip,tensorarray3d op[][][])
	{
		for(int i=0;i<ip.dim1;i++)
		{
			for(int j=0;j<ip.dim2;j++)
			{
				for(int k=0;k<ip.dim3;k++)
				{
					op[i][j][k].arr[0][0][0]=ip.arr[i][j][k];
				}
			}
		}
	}

	public tensorarray3d pad(tensorarray3d ip,int dim1,int dim2)
	{
		tensorarray3d op=new tensorarray3d(dim1,dim2,ip.dim3,ip.trainable);
		op.zeros();
		int diffrows=dim1-ip.dim1;
		int diffcols=dim2-ip.dim2;
		for(int i=0;i<ip.dim1;i++)
		{
			for(int j=0;j<ip.dim2;j++)
			{
				for(int k=0;k<ip.dim3;k++)
				{
					op.arr[i+diffrows][j+diffcols][k]=ip.arr[i][j][k];
				}
			}
		}
		return op;
	}
	public tensorarray3d getslices(tensorarray3d ip,int rowstart,int rowend,int colstart,int colend)
	{
		tensorarray3d op=new tensorarray3d(rowend-rowstart,colend-colstart,ip.dim3,ip.trainable);
		int numrows=rowend-rowstart;
		int numcolumns=colend-colstart;
		//System.out.println(numcolumns);
		//System.out.println(numrows);
		for(int i=0;i<numrows;i++)
		{
			for(int j=0;j<numcolumns;j++)
			{
				for(int k=0;k<ip.dim3;k++)
				{
					op.assigntensor(ip.arr[rowstart+i][colstart+j][k],i,j,k);
					
				}
			}
		}
		
		return op;
	}

}
