import tensordef.tensorgraph;
import tensordef.tensorarray;
import tensordef.add_tensorarray;
import tensordef.mul_tensorarray;
import tensordef.dot;
import tensordef.sigmoid;
import tensordef.tensorarray3d;
import tensordef.convolution;


public class convexample
{
	public static void main(String args[])
	{
		tensorgraph mod=new tensorgraph();
		tensorarray3d t1=new tensorarray3d(25,25,3,true);
		System.out.println(t1.arr[0][0][0].grad);
		convolution conv=new convolution(t1,3,1,mod,"SAME");
		tensorarray3d eval1=conv.forwardconv();
		mod.backward();
		System.out.println(t1.arr[0][0][0].grad);
	}
}
