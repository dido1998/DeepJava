package tensordef;
import basicops.*;
//import tensordef.*;
import java.util.*;


public class tensorgraph 
{
	ArrayList<backpropagationstructure<?>> oplist;
	public tensorgraph()
	{
		oplist=new ArrayList<backpropagationstructure<?>>();
	}
	public <T extends superopdef>void addtolist(backpropagationstructure<T> ob)
	{
		oplist.add(ob);
	}

	public <T extends superopdef> void removefromlist(backpropagationstructure<T> ob)
	{
		oplist.remove(ob);
	}
	public void backward()
	{
		
		backpropagationstructure<?> temp=oplist.get(oplist.size()-1);
		if(temp.ans!=null)
		{
			for(int i=0;i<temp.ans.dim1;i++)
			{
				for(int j=0;j<temp.ans.dim2;j++)
				{
					temp.ans.arr[i][j].grad=1;
				}
			}
		}else
		{
			for(int i=0;i<temp.ans1.dim1;i++)
			{
				for(int j=0;j<temp.ans1.dim2;j++)
				{
					for(int k=0;k<temp.ans1.dim3;k++)
					{
						temp.ans1.arr[i][j][k].grad=1;
					}
				}
			}
		}


		while(!oplist.isEmpty())
		{
			backpropagationstructure<?> curobjectref=oplist.get(oplist.size()-1);
			if(curobjectref.ans!=null)
				curobjectref.ob.backward(curobjectref.ans);
			else
			{
				//System.out.println(curobjectref.ob);
				curobjectref.ob.backwardconv(curobjectref.ans1);
			}
		}
	}
}
