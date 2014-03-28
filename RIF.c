/*Incomplete LDL^T factorization using Robust Incomplete Factorization*/

#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include <string.h>


double signfun(double x)/*sign of x*/
{
    return (x>=0) - (x<0);
}

double absval(double x)/*absolute value of x*/
{
    return x * signfun(x);
}


int eyeinit(size_t n, mwIndex *zcolstart, mwIndex *zcolend, 
        mwIndex *zrow, double *zval, mwIndex gap)/*Construct identity matrix CSC*/
{
    mwIndex j, next;
    
    for(j=0;j<gap && j<n;j++)
    {
        if(j > 0)next = next + j + 1;
        else
            next = 0;
        zcolstart[j] = next;
        zcolend[j] = next;
        zrow[next] = j;
        zval[next] = 1;
    }
    
    for(j>=gap;j<n;j++)
    {
        next = next + gap;
        zcolstart[j] = next;
        zcolend[j] = next;
        zrow[next] = j;
        zval[next] = 1;/*preallocate positions for gap+1:n*/
    }
    return 0;
}



/*Compute Matrix Vector Multiplication*/
int matxvec(size_t n,mwIndex *ajc,mwIndex *air,
        double *apr,mwIndex *zcolstart,mwIndex *zcolend,
        mwIndex *zrow,double *zval,mwIndex j,double *tmpAzj)
{
    mwIndex col,iter1,iter2,row;
    double val;/*compute Azj*/    
    for(iter1 = zcolstart[j];iter1<=zcolend[j];iter1++)
    {
        col = zrow[iter1];
        val = zval[iter1];
        for(iter2 = ajc[col];iter2 < ajc[col+1];iter2++)
        {
            row = air[iter2];
            tmpAzj[row] += apr[iter2] * val;
        }
    }
    return 0;
}



int addspvecs(mwIndex *zcolstart,mwIndex *zcolend,mwIndex *zrow,
        double *zval,double *sumvecval,mwIndex i,mwIndex j, double lambda,double alpha)/*add two sparse vectors zi + lambdazj*/
{
    mwIndex row_coli,row_colj,max;
    mwIndex iter_coli,iter_colj,tmpind;
    int iter;
    double sumvecnorm = 0;
    if(i<=j)
    {
        printf("i must be greater than j\n");
        error(1);
    }
    max = 0;
    
    for(iter_coli = zcolstart[i];iter_coli<=zcolend[i];iter_coli++)
    {
        row_coli = zrow[iter_coli];
        if(max<row_coli)max = row_coli;
        sumvecval[row_coli] = zval[iter_coli];
    }
    for(iter_colj=zcolstart[j];iter_colj<=zcolend[j];iter_colj++)
    {
        row_colj = zrow[iter_colj];
        if(max<row_colj)max = row_colj;
        sumvecval[row_colj] += lambda * zval[iter_colj];
    }
    
    for(iter=i;iter>=0;iter--)
    {
        sumvecnorm += absval(sumvecval[iter]);/*norm of resulting vector*/
    }
    
    
    tmpind = zcolend[i];
    for(iter=i;iter>=0;iter--)
    {
        if(absval(sumvecval[iter]) > alpha * sumvecnorm)
        {
            
            zrow[tmpind] = iter;
            zval[tmpind] = sumvecval[iter];            
            tmpind--;
        }
        if(tmpind<=zcolend[i-1])break;
    }
    zcolstart[i] = tmpind + 1;
    
    return 0;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, 
        const mxArray *prhs[])
{   
    /*declaration of variables*/
    /*iteration variables*/
    mwIndex i,j,k,numofnz;
    
    /*Auxillary variables*/
    double tmplij;
    mwIndex tmprowind;
    
    /*cputime variables*/
    clock_t start_time,end_time;
    double cputime;
    
    /*Step 2:*/
    size_t m, n;/*m:number of rows, n:number of columns*/
    mwSize nzmax;/*nnz of input*/
    
    size_t rnzmax=10000000;/*rnzmax should be comparable with nzmax*/
    
    mwIndex *ajc, *air;
    double *apr;/*CSC format of prhs[0]*/

    double gamma,tao, alpha,gapinput;/* gamma is the shift, and tao is threshold*/
    double *threshold,tmpthre;
    
    /*Step 3: */
    mwIndex *zcolstart, *zcolend, *zrow;
    double *zval;
    mwIndex gap = 1000;
    
    /*Step 4:*/
    double pjj,pij,lambda,normAzj;
    double  *tmpAzj;
    double *sumvecval;
    
    /*Output:*/    
    mwIndex *ljc, *lir;
    double *lpr;
    /*---------------------------------------------*/
    
    /* Step 1: Check input---------*/
    if(nrhs !=5)
    {
        mexErrMsgTxt("Five Input Arguments are required");
    }
    
    if(nlhs != 1)
        mexErrMsgTxt("One Output Argument Required\n");
        
    /*Check whether prhs[] is sparse*/
    if(mxIsSparse(prhs[0]) != 1)
        mexErrMsgTxt("Input Matrix Must Be Sparse\n");
    /* ------------------------------*/
    
    
    /* Step 2: Get Input Values------------*/
    /*Read m and n from prhs[0]*/
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    nzmax = mxGetNzmax(prhs[0]);
    
    /*CSC format of A=prhs[0]*/
    ajc = mxGetJc(prhs[0]);
    air = mxGetIr(prhs[0]);
    apr = mxGetPr(prhs[0]);
       
    
    /* Get shift*/
    gamma = mxGetScalar(prhs[1]);
    
    /*Get threshold*/
    tao = mxGetScalar(prhs[2]);
    threshold = mxCalloc(n,sizeof(double));
    
    for(j=0;j<n;j++)
    {
        for(i=ajc[j];i<ajc[j+1];i++)threshold[j] += absval(apr[i]);
        threshold[j] = tao * threshold[j];
    }
    
    /*Get threshold parameter for Z*/
    alpha = mxGetScalar(prhs[3]);
    
    /* Get allocation parameter*/
	gapinput = mxGetScalar(prhs[4]);
	gap = (mwIndex)gapinput;
	if(gap > n)gap = n;
    
    rnzmax = (size_t) (gap + 1) * n;
    
    /*---------------------------------------*/
    

    /*Step 3: Initialization of Z and L----------    */
    zcolstart = mxCalloc(n, sizeof(mwIndex));
    zcolend = mxCalloc(n, sizeof(mwIndex));
    zrow = mxCalloc(rnzmax,sizeof(mwIndex));
    zval = mxCalloc(rnzmax,sizeof(double));
    
    eyeinit(n, zcolstart, zcolend, zrow, zval, gap);/* Let Z be eye(n,n)*/
    
    /*---------------------------------------*/
    
    
    /* Step : Output    */
    plhs[0] = mxCreateSparse(n,n,rnzmax,mxREAL);    
    ljc = mxGetJc(plhs[0]);
    lir = mxGetIr(plhs[0]);
    lpr = mxGetPr(plhs[0]);
    
    
    
    /*Step 4: Compute L*/    
    numofnz = 0;
    tmpAzj = mxCalloc(m,sizeof(double));
    sumvecval = mxCalloc(n,sizeof(double));
    for(j=0;j<n;j++)
    {
        
        pjj = 0;
        memset(tmpAzj, 0, m*sizeof(double));
        matxvec(n,ajc,air,apr,zcolstart,zcolend,zrow,zval,j,tmpAzj);
        for(k=0;k<m;k++)
        {
            if(tmpAzj[k] != 0)pjj += tmpAzj[k] * tmpAzj[k];
        }
        pjj = pjj - pow(gamma,2.0) * zval[zcolend[j]];
        ljc[j] = numofnz;
        lir[numofnz] = j;
        lpr[numofnz] = sqrt(absval(pjj));
        numofnz = numofnz + 1;
        if(pjj < 2.2e-16){
            lpr[numofnz-1] = (threshold[j]>0)?threshold[j]:2.2e-16;
            continue;
        }
        for(i=j+1;i<n;i++)
        {
            pij = 0;
            for(k=ajc[i];k<ajc[i+1];k++)
            {
                tmprowind = air[k];
            	pij += apr[k] * tmpAzj[tmprowind];
            }
            /*zij = 0*/
            lambda = pij/pjj;
            if(absval(pij/sqrt(pjj)) > threshold[i])
            {
                lir[numofnz] = i;
                lpr[numofnz] = lambda * sqrt(pjj);
                numofnz = numofnz + 1;
                memset(sumvecval,0,n*sizeof(double));
                addspvecs(zcolstart,zcolend,zrow,zval,sumvecval,i,j,-lambda,alpha);
            }
        }
        
    }
    mxFree(tmpAzj);
    mxFree(sumvecval);
    mxFree(zcolstart);mxFree(zcolend);mxFree(zrow);mxFree(zval);/*free space*/
    ljc[n] = numofnz;
}
