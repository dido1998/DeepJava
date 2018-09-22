import tensordef.tensorgraph;
import tensordef.tensorarray;
import tensordef.add_tensorarray;
import tensordef.mul_tensorarray;
import tensordef.dot;
import tensordef.sigmoid;




public class fullyconnectedexample
{
	public static void main(String args[])
	{
		tensorgraph mod=new tensorgraph();
		tensorarray t1=new tensorarray(25,20,false);
		tensorarray w1=new tensorarray(20,10,true);
		tensorarray eval1=new tensorarray(25,10,false);
		tensorarray eval2=new tensorarray(25,10,false);
		System.out.println(t1.arr[0][0].grad);
		dot fc1=new dot(t1,w1,mod);
		eval1=fc1.forward();
		sigmoid act1=new sigmoid(eval1,mod);
		eval2=act1.forward();
		mod.backward();
		System.out.println(t1.arr[0][0].grad);
	}
}
