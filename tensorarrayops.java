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

	public tensorarray[] convert3dto2d(tensorarray3d arr)
	{
		tensorarray a[]=new tensorarray[arr.dim3];
		for(int i=0;i<arr.dim3;i++)
		{
			a[i]=new tensorarray(arr.dim1,arr.dim2,arr.trainable);
		}
		for(int i=0;i<arr.dim3;i++)
		{
			for(int j=0;j<arr.dim1;j++)
			{
				for(int k=0;k<arr.dim2;k++)
				{
					a[i].arr[j][k]=arr.arr[j][k][i];
				}
			}
		}
		return a;
	}
	public tensorarray stretch(tensorarray3d a,boolean rowise)
	{
		tensorarray b;
		if(rowise)
			b=new tensorarray(a.dim1*a.dim2*a.dim3,1,a.trainable);
		else
			 b=new tensorarray(1,a.dim1*a.dim2*a.dim3,a.trainable);
		int c =0;
		for(int k=0;k<a.dim3;k++)
		{
		for(int i=0;i<a.dim1;i++)
		{
			for(int j=0;j<a.dim2;j++)
			{
				if(rowise)
					b.arr[c++][0]=a.arr[i][j][k];
				else
					b.arr[0][c++]=a.arr[i][j][k];
			}
		}
		}
		return b;
	}
	public tensorarray3d convert2dto3d(tensorarray a[])
	{
		tensorarray3d b=new tensorarray3d(a[0].dim1,a[0].dim2,a.length,a[0].trainable);
		for(int i=0;i<b.dim1;i++)
		{
			for(int j=0;j<b.dim2;j++)
			{
				for(int k=0;k<b.dim3;k++)
				{
					b.arr[i][j][k]=a[k].arr[i][j];
				}
			}
		}
		return b;
	}
	public void convertotensorarray(tensorarray ip[][],tensorarray b,boolean trainable)
	{

		for(int i=0;i<ip.length;i++)
		{
			for(int j=0;j<ip[0].length;j++)
			{
				b.arr[i][j].data=ip[i][j].arr[0][0].data;
			}
		}
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
	public void tensorarray3dtoarrayoftensorarray2d(tensorarray3d a,tensorarray b[][][])
	{
		for(int i=0;i<a.dim1;i++)
		{
			for(int j=0;j<a.dim2;j++)
			{
				for(int k=0;k<a.dim3;k++)
				{
					b[i][j][k].arr[0][0]=a.arr[i][j][k];
				}
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
					//System.out.println(i+" "+j+" "+" "+k+" "+op[i][j][k]);
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
