import tensordef.tensorgraph;
import tensordef.tensorarray;
import tensordef.add_tensorarray;
import tensordef.mul_tensorarray;
import tensordef.dot;
import tensordef.sigmoid;
import tensordef.tensorarray3d;
import tensordef.convolution;


public class test
{
	public static void main(String args[])
	{
		tensorgraph mod=new tensorgraph();
		tensorarray3d t1=new tensorarray3d(25,25,3,true);
		//tensorarray t2=new tensorarray(5,4,true);
		//tensorarray t3=new tensorarray(4,4,true);
		/*double a2[][]={{-0.1,0.2,0.3,0.4},{0.1,0.2,0.3,0.4},{0.1,0.2,0.3,0.4},{0.1,0.2,0.3,0.4},{0.1,0.2,0.3,0.4}};
		double b[][]={{-0.1,0.2,0.3,0.4,0.5},{0.1,0.2,0.3,0.4,0.5},{0.1,0.2,0.3,0.4,0.5},{0.1,0.2,0.3,0.4,0.5} };
		double c[][]={ {0.2,0.2,0.2,0.2},{0.3,0.3,0.3,0.3}, {0.4,0.4,0.4,0.4} , {0.5,0.5,0.5,0.5} };
		t1.assign(b);
		t2.assign(a2);
		t3.assign(c);*/
		System.out.println(t1.arr[0][0][0].grad);
		convolution conv=new convolution(t1,5,3,mod,"SAME");
		tensorarray3d eval1=conv.forwardconv();
		mod.backward();
		/*for(int i=0;i<4;i++)
		{
			for(int j=0;j<4;j++)
			{
				System.out.println(t1.arr[i][j].grad);
			}
		}*/
		System.out.println(t1.arr[0][0][0].data);
	}
}
