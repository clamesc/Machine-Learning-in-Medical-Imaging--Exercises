// AUTHOR: Olivier Pauly
// EMAIL: pauly@cs.tum.edu

#include "mex.h"
#include "math.h"


// Principal MEX function
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray*prhs[] )
        
{

	// Inputs: Xdata and Nodes, Thresholds, Functions and Priors 
	// these 4 last inputs are attributes from the Tree object and have to be converted from cell to matrices 
	
	
	// First check number of data and dimensionality of data
	int N = mxGetM(prhs[0]);
	int InputDim = mxGetN(prhs[0]);
	
	// Then check output dimensionality
	int NbNodes = mxGetM(prhs[3]);
	int OutputDim = mxGetN(prhs[3])-2;
	
	// get a pointer on Xdata and the different tree attributes
	float *Xdata = (float*)mxGetData(prhs[0]);
	double *Nodes = (double*)mxGetData(prhs[1]);
	double *Thresholds = (double*)mxGetData(prhs[2]);
	double *Functions = (double*)mxGetData(prhs[3]);
	double *Priors = (double*)mxGetData(prhs[4]);
	
	// create output arrays
	plhs[0] = mxCreateNumericMatrix(N,OutputDim, mxSINGLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(N,1, mxDOUBLE_CLASS, mxREAL);
	
	// get output pointers
	float *Y = (float*)mxGetData(plhs[0]);
	double *P = (double*)mxGetData(plhs[1]);
	
	// now we can start with the main loop
	
	for(int i=0; i<N; i++)
	{
	
		// initialize tree index
		int index = 0;
		
		// go through the tree structure
		while(index>=0)
		{
			
			// go to the node and get the threshold
			int current_node = (int)Nodes[index]-1;
			double current_thr = Thresholds[index];
			
			if(current_node<0) // we reached a leaf
			{
				// store priors and function
				P[i] = Priors[index];
				for(int j=0; j<OutputDim; j++)
				{
					Y[i + j*N] = Functions[index + NbNodes*j]; 
		
				}
				index = -1;
			}
			else
			{
				// get test value and perform test
				float test_value = Xdata[i + current_node*N];
			
				if(test_value<current_thr)
				{
					index = (int)Nodes[index + NbNodes]-1;
				}
				else
				{
					index = (int)Nodes[index+ 2*NbNodes]-1;
				}
			
			}
			
		}
	
	
	}

}