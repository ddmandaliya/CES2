// $ make PETSC_DIR=/home/ddm/petsc PETSC_ARCH=linux-gnu-mpi FDM_2DSphere_NonIso_Ext
// $ ./FDM_2DSphere_NonIso_Ext -mat_type seqdense -pc_type sor -snes_converged_reason -ksp_converged_reason
// Edit makefile  

#include <math.h>
#include <stdio.h>
#include <petscsnes.h>
#include "petsc.h"
#define RxnOrder 2.0
#define phiG 9.0  //PhiG^2
#define NGrid 100
#define NGridP1 101
#define NGridP12 NGridP1*NGridP1
#define Shm -10000000000.0
#define Shh -10000000000.0
#define gama 10.0
#define beta 0.0
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian1(SNES,Vec,Mat,Mat,void*);
typedef struct
{
	double   *A;
	double   *B;
	double   **C;
	double   **D;
	double   **E;
	double   *rj;
	double   *mui;
} ApplicationCtx;
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	SNES           snes;         /* nonlinear solver */
	KSP            ksp;          /* linear solver */
	PC             pc;           /* preconditioner*/
	Vec            x,r;          /* solution, residual vectors */
	Mat            J;            /* Jacobian matrix */
	PetscErrorCode ierr;
	PetscInt       its;
	PetscMPIInt    size;
	PetscScalar    *xx;
	PetscBool      flg;
	ApplicationCtx ctx;
	PetscInitialize(&argc,&argv,(char*)0,help);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Example is only for sequential runs");
	PetscLogDouble v1,v2,elapsed_time;
	ierr = PetscGetCPUTime(&v1);CHKERRQ(ierr);
	//	omp_set_num_threads(NUM_THREADS);
	ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
	ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,(2*NGridP1*NGridP1));CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
	int i=0, j=0, f;
	ctx.C = (double**) malloc((NGridP1)*sizeof(double*));
    	if (ctx.C)
    	{
		for (i = 0; i < NGridP1; i++)
		{
			ctx.C[i]= (double*) malloc((NGridP1)*sizeof(double));
			if (ctx.C[i] == NULL)
			{
				for (f = i; f >= 0; f--)
				{
					if (ctx.C[f] != NULL)
					{
						free(ctx.C[f]);
						ctx.C[f] = NULL;
					}
				}
				if (ctx.C != NULL)
				{
					free(ctx.C);
					ctx.C = NULL;
				}
				return 0;
			}
		}
    	}
    	else
    	{
		return 0;
    	}
	ctx.A= (double*) malloc((NGridP1)*sizeof(double));
	if (ctx.A == NULL)
	{
		free(ctx.A);
		ctx.A = NULL;
	}
	ctx.B= (double*) malloc((NGridP1)*sizeof(double));
	if (ctx.B == NULL)
	{
		free(ctx.B);
		ctx.B = NULL;
	}
	ctx.D = (double**) malloc((NGridP1)*sizeof(double*));
    	if (ctx.D)
    	{
        	for (i = 0; i < NGridP1; i++)
		{
			ctx.D[i]= (double*) malloc((NGridP1)*sizeof(double));
			if (ctx.D[i] == NULL)
			{
				for (f = i; f >= 0; f--)
				{
					if (ctx.D[f] != NULL)
					{
						free(ctx.D[f]);
						ctx.D[f] = NULL;
					}
				}
				if (ctx.D != NULL)
				{
					free(ctx.D);
					ctx.D = NULL;
				}
				return 0;
			}
		}
    	}
	else
	{
		return 0;
	}

	ctx.E = (double**) malloc((NGridP1)*sizeof(double*));
	if (ctx.E)
	{
		for (i = 0; i < NGridP1; i++)
		{
			ctx.E[i]= (double*) malloc((NGridP1)*sizeof(double));
			if (ctx.E[i] == NULL)
			{
				for (f = i; f >= 0; f--)
				{
					if (ctx.E[f] != NULL)
					{
						free(ctx.E[f]);
						ctx.E[f] = NULL;
					}
				}
				if (ctx.E != NULL)
				{
					free(ctx.E);
					ctx.E = NULL;
				}
				return 0;
			}
		}
	}
    	else
    	{
		return 0;
	}
    	double delr=1.0/((double)NGrid);
	double delmu=2*delr;
	ctx.rj= (double*) malloc((NGridP1)*sizeof(double));
	if (ctx.rj == NULL)
	{
		free(ctx.rj);
		ctx.rj = NULL;
	}
    	ctx.mui= (double*) malloc((NGridP1)*sizeof(double));
	if (ctx.mui == NULL)
	{
		free(ctx.mui);
		ctx.mui = NULL;
	}
	for (j = 0; j < (NGrid+1); j++)
    	{
		ctx.rj[j] =1.0*(j*((1.0/(double)(NGrid))));
		ctx.mui[j] =1.0- 1.0*(j*(2.0/(double)(NGrid)));
	}
	j=0;
	ctx.A[0] = 3.0/(pow(delr,2));
	ctx.B[0] = 3.0/(pow(delr,2));
	for (i = 0; i < (NGrid+1); i++)
    	{
		ctx.C[0][i] =-6.0/(pow(delr,2));
	}
	for (j = 1; j <= (NGrid-1); j++)
	{
		ctx.A[j] = (1.0/(pow(delr,2)))*(1.0+(1.0/(double)j));
		ctx.B[j] = (1.0/(pow(delr,2)))*(1.0-(1.0/(double)j));
	    	for (i = 0; i < (NGrid+1); i++)
        	{
			if(i==0)
			{
				double term1=2.0/(pow(delr,2));
				double term2=4.0/(pow(delmu*ctx.rj[j],2));
				ctx.C[j][i] =-1.0*(term1+term2);
				ctx.D[j][i] = term2/2.0;
				ctx.E[j][i] = term2/2.0;
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			if(i==NGrid)
			{
				double term1=2.0/(pow(delr,2));
				double term2=4.0/(pow(delmu*ctx.rj[j],2));
				ctx.C[j][i] =-1.0*(term1+term2);
				ctx.D[j][i] = term2/2.0;
				ctx.E[j][i] = term2/2.0;
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			if(i==(int)(NGrid/2))
			{
				ctx.C[j][i] =-2.0/(pow(delr,2));
				ctx.D[j][i] = 0.0;
				ctx.E[j][i] = 0.0;
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			else
			{
				double term1=2.0/(pow(delr,2));
				double term2=(1.0-pow(ctx.mui[i],2))/(pow(delmu*ctx.rj[j],2));
				double term3=ctx.mui[i]/(delmu*ctx.rj[j]);
				ctx.C[j][i] =-1.0*(term1+2*term2);
				ctx.D[j][i] = (term2-term3);
				ctx.E[j][i] = (term2+term3);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
        	}
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t A[%D]= %g",j,j,ctx.A[j]);CHKERRQ(ierr);
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t B[%D]= %g",j,j,ctx.B[j]);CHKERRQ(ierr);
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t rj[%D]= %g",j,j,ctx.rj[j]);CHKERRQ(ierr);
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t mui[%D]= %g",j,j,ctx.mui[j]);CHKERRQ(ierr);
	}
		j=NGrid;
		ctx.A[j] = 0.0;
		ctx.B[j] = (2.0/(Shm*pow(delr,2)));
		for (i = 0; i < (NGrid+1); i++)
        	{
			if(i==0)
			{
				ctx.C[j][i] =(2.0/delr)+(2.0/ctx.rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(Shm*pow((delmu*ctx.rj[j]),2)));
				ctx.D[j][i] = (2.0/(Shm*pow((delmu*ctx.rj[j]),2)));
				ctx.E[j][i] = (2.0/(Shm*pow((delmu*ctx.rj[j]),2)));
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			if(i==NGrid)
			{
				ctx.C[j][i] =(2.0/delr)+(2.0/ctx.rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(pow((delmu*ctx.rj[j]),2)));
				ctx.D[j][i] = (2.0/(Shm*pow((delmu*ctx.rj[j]),2)));
				ctx.E[j][i] = (2.0/(Shm*pow((delmu*ctx.rj[j]),2)));
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			if(i==(int)(NGrid/2))
			{
				ctx.C[j][i] =(2.0/delr)+(2.0/ctx.rj[j])-(2.0/(Shm*pow(delr,2)));
				ctx.D[j][i] = 0.0;
				ctx.E[j][i] = 0.0;
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
			else
			{
				double term1=(1.0-pow(ctx.mui[i],2))/(pow(delmu*ctx.rj[j],2));
				double term2=ctx.mui[i]/(delmu*ctx.rj[j]);
				ctx.C[j][i] =(2.0/delr)+(2.0/ctx.rj[j])-(2.0/(Shm*pow(delr,2)))-(2.0*term1/Shm);
				ctx.D[j][i] = (term1-term2)/Shm;
				ctx.E[j][i] = (term1+term2)/Shm;
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  C[%D][%D]= %g",j,i,j,i,ctx.C[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  D[%D][%D]= %g",j,i,j,i,ctx.D[j][i]);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t i=%D \t  E[%D][%D]= %g",j,i,j,i,ctx.E[j][i]);CHKERRQ(ierr);
			}
        	}
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t A[%D]= %g",j,j,ctx.A[j]);CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t B[%D]= %g",j,j,ctx.B[j]);CHKERRQ(ierr);
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t rj[%D]= %g",j,j,ctx.rj[j]);CHKERRQ(ierr);
        	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n j=%D  \t mui[%D]= %g",j,j,ctx.mui[j]);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
	ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,(2*NGridP1*NGridP1),(2*NGridP1*NGridP1));CHKERRQ(ierr);
	ierr = MatSetFromOptions(J);CHKERRQ(ierr);
	ierr = MatSetUp(J);CHKERRQ(ierr);
	ierr = PetscOptionsHasName(NULL,NULL,"-hard",&flg);CHKERRQ(ierr);
	ierr = SNESSetFunction(snes,r,FormFunction1,&ctx);CHKERRQ(ierr);
	ierr = SNESSetJacobian(snes,J,J,FormJacobian1,&ctx);CHKERRQ(ierr);
	ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
	ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
	ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20);CHKERRQ(ierr);
	ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
	PetscInt idxin[2*NGridP1*NGridP1];
	PetscScalar xval[2*NGridP1*NGridP1];
	for(int iinv=0;iinv<(2*NGridP1*NGridP1);iinv=iinv+1)
	{
		idxin[iinv]=iinv;
	}
	if (!flg)
	{
		for(int xiniv=0;xiniv<(NGridP1*NGridP1);xiniv++)
		{
			xval[xiniv]=(double)xiniv/((double)NGridP1*NGridP1);
			ierr = PetscPrintf(PETSC_COMM_WORLD,"Guess Value Conc. %D = %g\n",xiniv,xval[xiniv]);CHKERRQ(ierr);
		}
		for(int xiniv=(NGridP1*NGridP1);xiniv<(2*NGridP1*NGridP1);xiniv++)
		{
			xval[xiniv]=(double)(xiniv-(NGridP1*NGridP1))/((double)NGridP1*NGridP1);
			ierr = PetscPrintf(PETSC_COMM_WORLD,"Guess Value Temp. %D = %g\n",xiniv,xval[xiniv]);CHKERRQ(ierr);
		}
		VecSetValues(x,(2*NGridP1*NGridP1),idxin,xval, INSERT_VALUES);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting SNES Solve ");CHKERRQ(ierr);
	ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"SNES Solve completed");CHKERRQ(ierr);
	ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);
	ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	PetscScalar    *array1;
	VecGetArray(x,&array1);
	for (int i1=0; i1<(2*NGridP1*NGridP1); i1++)
	{
		PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%D %.50g\n",i1,(double)PetscRealPart(array1[i1]));
	}
	VecRestoreArray(x,&array1);
	PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
	ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&r);CHKERRQ(ierr);
	ierr = MatDestroy(&J);CHKERRQ(ierr); ierr = SNESDestroy(&snes);CHKERRQ(ierr);
	//free(ctx.TestOrder);
	for (i = 0; i < (NGrid+1); i++)
	{
		free(ctx.C[i]);
		ctx.C[i] = NULL;
	}
	free(ctx.C);
    	ctx.C = NULL;
    	free(ctx.A);
    	ctx.A = NULL;
    	free(ctx.B);
    	ctx.B = NULL;
    	free(ctx.D);
    	ctx.D = NULL;
    	free(ctx.E);
    	ctx.E = NULL;
    	free(ctx.rj);
    	ctx.rj = NULL;
    	free(ctx.mui);
    	ctx.mui = NULL;
	ierr = PetscGetCPUTime(&v2);CHKERRQ(ierr);
	elapsed_time = v2 - v1;
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Elaspsed Time = %g\n",elapsed_time);CHKERRQ(ierr);
 	ierr = PetscFinalize();
  	return 0;
}
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *ctx)
{
	PetscErrorCode    ierr;
	const PetscScalar *xx;
	PetscScalar       *ff;
	ApplicationCtx *user = (ApplicationCtx*) ctx;
	ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
	ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
	PetscInt j=0,i=0;
	double delr=1.0/((double)NGrid);
	double delmu=2*delr;
	for(i=0;i<=NGrid;i++)
	{
		if(i==(int)(NGrid/2))
		{
			ff[i]=((user->A[0])+(user->B[0]))*xx[NGridP1*1+i]+(user->C[0][i])*xx[i]-phiG*pow(xx[i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+i])));
		}
		if(i<(int)(NGrid/2))
		{
			ff[i]=xx[i]-xx[i+1];
		}
		if(i>(int)(NGrid/2))
		{
			ff[i]=xx[i]-xx[i-1];
		}
	}
	for (j = 1; j < (NGrid); j++)
    	{
		for (i = 0; i < NGridP1; i++)
        	{
			if(i==0)
			{
				ff[NGridP1*j+i]=((user->A[j])*xx[NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP1*(j-1)+i])+((user->C[j][0])*xx[NGridP1*j+i])+((user->D[j][0])*xx[NGridP1*j+1])+((user->E[j][0])*xx[NGridP1*j+1])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i==NGrid)
			{
				ff[NGridP1*j+i]=((user->A[j])*xx[NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP1*j+i])+((user->D[j][i])*xx[NGridP1*j+(NGrid-1)])+((user->E[j][i])*xx[NGridP1*j+(NGrid-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i==(int)(NGrid/2))
			{
				ff[NGridP1*j+i]=((user->A[j])*xx[NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP1*j+i])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
			{
				ff[NGridP1*j+i]=((user->A[j])*xx[NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP1*j+i])+((user->D[j][i])*xx[NGridP1*j+i+1])+((user->E[j][i])*xx[NGridP1*j+(i-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
        	}
	}
	j=NGrid;
	for (i = 0; i < NGridP1; i++)
	{
		if(i==0)
		{
			ff[NGridP1*j+i]=((2.0/(Shm*pow(delr,2)))*xx[NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(Shm*pow((delmu*user->rj[j]),2))))*xx[NGridP1*j+i])+((2.0/(Shm*pow((delmu*user->rj[j]),2)))*xx[NGridP1*j+1])+((2.0/(Shm*pow((delmu*user->rj[j]),2)))*xx[NGridP1*j+1])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i==NGrid)
		{
			ff[NGridP1*j+i]=((2.0/(Shm*pow(delr,2)))*xx[NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(Shm*pow((delmu*user->rj[j]),2))))*xx[NGridP1*j+i])+((2.0/(Shm*pow((delmu*user->rj[j]),2)))*xx[NGridP1*j+(NGrid-1)])+((2.0/(Shm*pow((delmu*user->rj[j]),2)))*xx[NGridP1*j+(NGrid-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i==(int)(NGrid/2))
		{
			ff[NGridP1*j+i]=((2.0/(Shm*pow(delr,2)))*xx[NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2))))*xx[NGridP1*j+i])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
		{
			double term1=(1.0-pow(user->mui[i],2))/(pow(delmu*user->rj[j],2));
			double term2=user->mui[i]/(delmu*user->rj[j]);
			ff[NGridP1*j+i]=((2.0/(Shm*pow(delr,2)))*xx[NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(2.0*term1/Shm))*xx[NGridP1*j+i])+(((term1-term2)/Shm)*xx[(NGridP1*j)+(i+1)])+(((term1+term2)/Shm)*xx[(NGridP1*j)+(i-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm)-(2.0/delr)-(2.0/user->rj[j]);
		}
	}
	for(i=0;i<=NGrid;i++)
	{
		if(i==(int)(NGrid/2))
		{
			ff[NGridP12+i]=((user->A[0])+(user->B[0]))*xx[NGridP12+NGridP1*1+i]+(user->C[0][i])*xx[NGridP12+i]-phiG*pow(xx[i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+i])));
		}
		if(i<(int)(NGrid/2))
		{
			ff[NGridP12+i]=xx[NGridP12+i]-xx[NGridP12+i+1];
		}
		if(i>(int)(NGrid/2))
		{
			ff[NGridP12+i]=xx[NGridP12+i]-xx[NGridP12+i-1];
		}
	}
	for (j = 1; j < (NGrid); j++)
    	{
	    	for (i = 0; i < NGridP1; i++)
        	{
			if(i==0)
			{
				ff[NGridP12+NGridP1*j+i]=((user->A[j])*xx[NGridP12+NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP12+NGridP1*(j-1)+i])+((user->C[j][0])*xx[NGridP12+NGridP1*j+i])+((user->D[j][0])*xx[NGridP12+NGridP1*j+1])+((user->E[j][0])*xx[NGridP12+NGridP1*j+1])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i==NGrid)
			{
				ff[NGridP12+NGridP1*j+i]=((user->A[j])*xx[NGridP12+NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP12+NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP12+NGridP1*j+i])+((user->D[j][i])*xx[NGridP12+NGridP1*j+(NGrid-1)])+((user->E[j][i])*xx[NGridP12+NGridP1*j+(NGrid-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i==(int)(NGrid/2))
			{
				ff[NGridP12+NGridP1*j+i]=((user->A[j])*xx[NGridP12+NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP12+NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP12+NGridP1*j+i])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
			if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
			{
				ff[NGridP12+NGridP1*j+i]=((user->A[j])*xx[NGridP12+NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP12+NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP12+NGridP1*j+i])+((user->D[j][i])*xx[NGridP12+NGridP1*j+i+1])+((user->E[j][i])*xx[NGridP12+NGridP1*j+(i-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
			}
        }
	}
	j=NGrid;
	for (i = 0; i < NGridP1; i++)
	{
		if(i==0)
		{
			ff[NGridP12+NGridP1*j+i]=((2.0/(Shh*pow(delr,2)))*xx[NGridP12+NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(4.0/(Shh*pow((delmu*user->rj[j]),2))))*xx[NGridP12+NGridP1*j+i])+((2.0/(Shh*pow((delmu*user->rj[j]),2)))*xx[NGridP12+NGridP1*j+1])+((2.0/(Shh*pow((delmu*user->rj[j]),2)))*xx[NGridP12+NGridP1*j+1])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i==NGrid)
		{
			ff[NGridP12+NGridP1*j+i]=((2.0/(Shh*pow(delr,2)))*xx[NGridP12+NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(4.0/(Shh*pow((delmu*user->rj[j]),2))))*xx[NGridP12+NGridP1*j+i])+((2.0/(Shh*pow((delmu*user->rj[j]),2)))*xx[NGridP12+NGridP1*j+(NGrid-1)])+((2.0/(Shh*pow((delmu*user->rj[j]),2)))*xx[NGridP12+NGridP1*j+(NGrid-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i==(int)(NGrid/2))
		{
			ff[NGridP12+NGridP1*j+i]=((2.0/(Shh*pow(delr,2)))*xx[NGridP12+NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2))))*xx[NGridP12+NGridP1*j+i])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh)-(2.0/delr)-(2.0/user->rj[j]);
		}
		if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
		{
			double term1=(1.0-pow(user->mui[i],2))/(pow(delmu*user->rj[j],2));
			double term2=user->mui[i]/(delmu*user->rj[j]);
			ff[NGridP12+NGridP1*j+i]=((2.0/(Shh*pow(delr,2)))*xx[NGridP12+NGridP1*(j-1)+i])+(((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(2.0*term1/Shh))*xx[NGridP12+NGridP1*j+i])+(((term1-term2)/Shh)*xx[NGridP12+NGridP1*j+i+1])+(((term1+term2)/Shh)*xx[NGridP12+NGridP1*j+(i-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh)-(2.0/delr)-(2.0/user->rj[j]);
		}
	}
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  return 0;
}
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
	PetscErrorCode    ierr;
	const PetscScalar *xx;
	PetscScalar       AJ[2*NGridP1*NGridP1];
	PetscInt          idx[2*NGridP1*NGridP1];
	ApplicationCtx *user = (ApplicationCtx*) dummy;
	PetscInt i,j,qrow=0;
	for(i=0;i<(2*NGridP1*NGridP1);i=i+1)
	{
		idx[i]=i;
	}
	ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
	double delr=1.0/((double)NGrid);
	double delmu=2*delr;
	for(i=0;i<=NGrid;i++)
	{
		for(int it=0;it<(2*NGridP1*NGridP1);it++)
		{
			AJ[it]=0.0;
		}

		if(i==(int)(NGrid/2))
		{
			AJ[NGridP1*1+i]=((user->A[0])+(user->B[0]));
			AJ[i]=(user->C[0][i])-phiG*RxnOrder*pow(xx[i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+i])));
			AJ[NGridP12+i]=-phiG*pow(xx[i],RxnOrder)*(gama/pow(xx[NGridP12+i],2))*exp(gama*(1-(1/xx[NGridP12+i])));
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i<(int)(NGrid/2))
		{
			AJ[i]=1.0;
			AJ[i+1]=-1.0;
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i>(int)(NGrid/2))
		{
			//ff[i]=xx[i]-xx[i-1];
			AJ[i]=1.0;
			AJ[i-1]=-1.0;
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		qrow=qrow+1;
	}

	for (j = 1; j < NGrid; j++)
    	{
	    	for (i = 0; i < NGridP1; i++)
        	{
			for(int it=0;it<(2*NGridP1*NGridP1);it++)
			{
				AJ[it]=0.0;
			}
			if(i==0)
			{
				AJ[NGridP1*(j+1)+i]=(user->A[j]);
				AJ[NGridP1*(j-1)+i]=(user->B[j]);
				AJ[NGridP1*j+i]=(user->C[j][0])-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP1*j+1]=user->D[j][0]+user->E[j][0];
				AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i==NGrid)
			{
				AJ[NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP1*j+i]=user->C[j][i]-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP1*j+(NGrid-1)]=user->D[j][i]+user->E[j][i];
				AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i==(int)(NGrid/2))
			{
				AJ[NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP1*j+i]=user->C[j][i]-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
			{
				//ff[NGridP1*j+i]=((user->A[j])*xx[NGridP1*(j+1)+i])+((user->B[j])*xx[NGridP1*(j-1)+i])+((user->C[j][i])*xx[NGridP1*j+i])+((user->D[j][i])*xx[NGridP1*j+i+1])+((user->E[j][i])*xx[NGridP1*j+(i-1)])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP1*j+i]=user->C[j][i]-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP1*j+i+1]=user->D[j][i];
				AJ[NGridP1*j+(i-1)]=user->E[j][i];
				AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			qrow=qrow+1;
        	}
	}
	j=NGrid;
	for (i = 0; i < NGridP1; i++)
	{
		for(int it=0;it<(2*NGridP1*NGridP1);it++)
		{
			AJ[it]=0.0;
		}
		if(i==0)
		{
			AJ[NGridP1*(j-1)+i]=(2.0/(Shm*pow(delr,2)));
			AJ[NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(Shm*pow((delmu*user->rj[j]),2))))-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			AJ[NGridP1*j+1]=(2.0/(Shm*pow((delmu*user->rj[j]),2)))+(2.0/(Shm*pow((delmu*user->rj[j]),2)));
			AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i==NGrid)
		{
			AJ[NGridP1*(j-1)+i]=(2.0/(Shm*pow(delr,2)));
			AJ[NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(4.0/(Shm*pow((delmu*user->rj[j]),2))))-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			AJ[NGridP1*j+(NGrid-1)]=(2.0/(Shm*pow((delmu*user->rj[j]),2)))+(2.0/(Shm*pow((delmu*user->rj[j]),2)));
			AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i==(int)(NGrid/2))
		{
			AJ[NGridP1*(j-1)+i]=(2.0/(Shm*pow(delr,2)));
			AJ[NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2))))-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
		{
			double term1=(1.0-pow(user->mui[i],2))/(pow(delmu*user->rj[j],2));
			double term2=user->mui[i]/(delmu*user->rj[j]);
			AJ[NGridP1*(j-1)+i]=(2.0/(Shm*pow(delr,2)));
			AJ[NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shm*pow(delr,2)))-(2.0*term1/Shm))-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			AJ[NGridP1*j+i+1]=((term1-term2)/Shm);
			AJ[NGridP1*j+(i-1)]=((term1+term2)/Shm);
			AJ[NGridP12+NGridP1*j+i]=-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shm);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		qrow=qrow+1;
	}
	for(i=0;i<=NGrid;i++)
	{
		for(int it=0;it<(2*NGridP1*NGridP1);it++)
		{
			AJ[it]=0.0;
		}
		if(i==(int)(NGrid/2))
		{
			AJ[NGridP12+NGridP1*1+i]=((user->A[0])+(user->B[0]));
			AJ[NGridP12+i]=(user->C[0][i])-phiG*pow(xx[i],RxnOrder)*(gama/pow(xx[NGridP12+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+i])));
			AJ[i]=-phiG*RxnOrder*pow(xx[i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+i])));
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i<(int)(NGrid/2))
		{
			AJ[NGridP12+i]=1.0;
			AJ[NGridP12+i+1]=-1.0;
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i>(int)(NGrid/2))
		{
			AJ[NGridP12+i]=1.0;
			AJ[NGridP12+i-1]=-1.0;
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		qrow=qrow+1;
	}
	for (j = 1; j < NGrid; j++)
    	{
	    	for (i = 0; i < NGridP1; i++)
        	{
			for(int it=0;it<(2*NGridP1*NGridP1);it++)
			{
				AJ[it]=0.0;
			}
			if(i==0)
			{
				AJ[NGridP12+NGridP1*(j+1)+i]=(user->A[j]);
				AJ[NGridP12+NGridP1*(j-1)+i]=(user->B[j]);
				AJ[NGridP12+NGridP1*j+i]=(user->C[j][0])-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP12+NGridP1*j+1]=user->D[j][0]+user->E[j][0];
				AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i==NGrid)
			{
				AJ[NGridP12+NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP12+NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP12+NGridP1*j+i]=user->C[j][i]-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP12+NGridP1*j+(NGrid-1)]=user->D[j][i]+user->E[j][i];
				AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i==(int)(NGrid/2))
			{
				AJ[NGridP12+NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP12+NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP12+NGridP1*j+i]=user->C[j][i]-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
			{
				AJ[NGridP12+NGridP1*(j+1)+i]=user->A[j];
				AJ[NGridP12+NGridP1*(j-1)+i]=user->B[j];
				AJ[NGridP12+NGridP1*j+i]=user->C[j][i]-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				AJ[NGridP12+NGridP1*j+i+1]=user->D[j][i];
				AJ[NGridP12+NGridP1*j+(i-1)]=user->E[j][i];
				AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i]))));
				ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
			}
			qrow=qrow+1;
        	}
	}
	j=NGrid;
	for (i = 0; i < NGridP1; i++)
	{
		for(int it=0;it<(2*NGridP1*NGridP1);it++)
		{
			AJ[it]=0.0;
		}
		if(i==0)
		{
			AJ[NGridP12+NGridP1*(j-1)+i]=(2.0/(Shh*pow(delr,2)));
			AJ[NGridP12+NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(4.0/(Shh*pow((delmu*user->rj[j]),2))))-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			AJ[NGridP12+NGridP1*j+1]=(2.0/(Shh*pow((delmu*user->rj[j]),2)))+(2.0/(Shh*pow((delmu*user->rj[j]),2)));
			AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i==NGrid)
		{
			AJ[NGridP12+NGridP1*(j-1)+i]=(2.0/(Shh*pow(delr,2)));
			AJ[NGridP12+NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(4.0/(Shh*pow((delmu*user->rj[j]),2))))-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			AJ[NGridP12+NGridP1*j+(NGrid-1)]=(2.0/(Shh*pow((delmu*user->rj[j]),2)))+(2.0/(Shh*pow((delmu*user->rj[j]),2)));
			AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i==(int)(NGrid/2))
		{
			AJ[NGridP12+NGridP1*(j-1)+i]=(2.0/(Shh*pow(delr,2)));
			AJ[NGridP12+NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2))))-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		if(i!=0&&i!=NGrid&&i!=(int)(NGrid/2))
		{
			double term1=(1.0-pow(user->mui[i],2))/(pow(delmu*user->rj[j],2));
			double term2=user->mui[i]/(delmu*user->rj[j]);
			AJ[NGridP12+NGridP1*(j-1)+i]=(2.0/(Shh*pow(delr,2)));
			AJ[NGridP12+NGridP1*j+i]=((2.0/delr)+(2.0/user->rj[j])-(2.0/(Shh*pow(delr,2)))-(2.0*term1/Shh))-(phiG*pow(xx[NGridP1*j+i],RxnOrder)*(gama/pow(xx[NGridP12+NGridP1*j+i],2))*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			AJ[NGridP12+NGridP1*j+i+1]=((term1-term2)/Shh);
			AJ[NGridP12+NGridP1*j+(i-1)]=((term1+term2)/Shh);
			AJ[NGridP1*j+i]=-(phiG*RxnOrder*pow(xx[NGridP1*j+i],RxnOrder-1)*beta*exp(gama*(1-(1/xx[NGridP12+NGridP1*j+i])))/Shh);
			ierr = MatSetValues(B,1,&qrow,(2*NGridP1*NGridP1),idx,AJ,INSERT_VALUES);CHKERRQ(ierr);
		}
		qrow=qrow+1;
	}
	ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	if (jac != B)
	{
		ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	}
	return 0;
}