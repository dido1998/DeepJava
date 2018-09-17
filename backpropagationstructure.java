package tensordef;
//import tensordef.*;
import java.util.*;

public class backpropagationstructure<T extends superopdef>
{

	T ob;
	tensorarray ans;
	tensorarray3d ans1;
	public backpropagationstructure(T ob,tensorarray ans,tensorarray3d ans1)
	{
		this.ob=ob;
		this.ans=ans;
		this.ans1=ans1;
	}
}