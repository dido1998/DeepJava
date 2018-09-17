package tensordef;
import basicops.*;
import java.math.*;
import java.util.*;
public class sigmoid extends superopdef
{
	tensorarray arr;
	tensorarray ones1;
	tensorarray ones2;
	add_tensorarray addition;
	div_tensorarray division;
	exp_tensorarray exponenent;
	negate_tensorarray negops;
	tensorarray eval0;
	tensorarray eval1;
	tensorarray eval2;
	tensorgraph graph;
	tensorarray eval;
	backpropagationstructure<sigmoid>  curstruct;
	public sigmoid(tensorarray arr1,tensorgraph graph)
	{
		arr=arr1;
		this.graph=graph;
		ones1=new tensorarray(arr.dim1,arr.dim2,false);
		ones1.ones();
		
		ones2=new tensorarray(arr.dim1,arr.dim2,false);
		ones2.ones();
		eval0=new tensorarray(arr.dim1,arr.dim2,false);		
		eval1=new tensorarray(arr.dim1,arr.dim2,false);
		eval2=new tensorarray(arr.dim1,arr.dim2,false);
		eval=new tensorarray(arr.dim1,arr.dim2,false);
		curstruct=new backpropagationstructure<sigmoid>(this,eval,null);
		graph.addtolist(curstruct);
		negops=new negate_tensorarray(arr,graph);
		exponenent=new exp_tensorarray(eval0,graph);
		addition=new add_tensorarray(ones1,eval1,graph);
		division=new div_tensorarray(ones2,eval2,graph);
		
	}

	public tensorarray forward()
	{
		eval0.assigntensorarray(negops.forward());
		eval1.assigntensorarray(exponenent.forward());
		eval2.assigntensorarray(addition.forward());
		eval.assigntensorarray(division.forward());
		return eval;
	}
	public void backward(tensorarray backflow)
	{
		division.backward(backflow);
		addition.backward(eval2);
		exponenent.backward(eval1);
		negops.backward(eval0);

		graph.removefromlist(curstruct);
	}
}
