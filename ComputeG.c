// $ gcc -Wall ComputeG.c -o ComputeG -lm -lgmp -lmpfr -fopenmp
// $ ./ComputeG

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>
#include <mpfr.h>
#include <omp.h>
#define NGrid 101
#define NUM_THREADS 32

int main() 
{
	int p,q,j,i;
	FILE *fp;
	fp = fopen ("ParallelizeStudy_outputComputeG_Rev06N101.txt", "a+");
	omp_set_num_threads(NUM_THREADS);
	float delrstep=(1.0/(NGrid-1));
	float delmustep=(1.0/(NGrid-1));
	unsigned int precd=128;
	mpfr_t t1,t2,t3,zi,eta,rp,muq;
	mpfr_inits2 (precd,t1,t2,t3,zi,eta,rp,muq,(mpfr_ptr) 0);
	int ii;
    	float zijF,etaiF,rpF,muqF;	
	mpfr_t *Pimu = malloc(1100 * sizeof(*Pimu));
	for(i = 0; i <= 1099; i++)
	{
		mpfr_inits2(precd,Pimu[i],(mpfr_ptr) 0);
	}
	mpfr_t *Pieta = malloc(1100 * sizeof(*Pieta));
	for(i = 0; i <= 1099; i++)
	{
		mpfr_inits2(precd,Pieta[i],(mpfr_ptr) 0);
	}	
	unsigned int n;				
	mpfr_t GmultiP;
	mpfr_inits2 (precd,GmultiP,(mpfr_ptr) 0);
	for (p = 0; p < NGrid; p++) 
    	{
        	for (q = 0; q < NGrid; q++)
        	{
            		for (j = 0; j < NGrid; j++) 
            		{
                		for (i = 0; i < NGrid; i++) 
                		{
					zijF=j*delrstep;
					etaiF=i*delmustep;
					rpF=p*delrstep;
					muqF=q*delmustep;
					mpfr_set_flt(zi, zijF,10);
					mpfr_set_flt (eta, etaiF,10);
					mpfr_set_flt( rp, rpF, 10);
					mpfr_set_flt( muq, muqF, 10);
					mpfr_set_str( Pimu[0], "1.0", 10, MPFR_RNDN);
					mpfr_set( Pimu[1], muq, 10);
					for(ii = 1; ii <= 1090; ii++)
					{
						mpfr_mul_ui(t1,Pimu[ii-1],ii,MPFR_RNDU);
						mpfr_set_str( t2, "2.0", 10, MPFR_RNDN);
						mpfr_mul_ui(t2,t2,ii,MPFR_RNDU);
						mpfr_add_ui(t2,t2,1,MPFR_RNDD);
						mpfr_mul(t2,t2,muq,MPFR_RNDN);
						mpfr_mul(t2,t2,Pimu[ii],MPFR_RNDU);
						mpfr_sub(t2,t2,t1,MPFR_RNDD);
						mpfr_set_ui(t3, ii, MPFR_RNDN);
						mpfr_add_ui(t3,t3,1,MPFR_RNDD);
						mpfr_div(t2,t2,t3,MPFR_RNDD);
						mpfr_set(Pimu[ii+1], t2, MPFR_RNDD);
					}
					mpfr_set_str( Pieta[0], "1.0", 10, MPFR_RNDN);
					mpfr_set( Pieta[1], eta, 10);
					for(ii = 1; ii <= 1090; ii++)
					{
						mpfr_mul_ui(t1,Pieta[ii-1],ii,MPFR_RNDU);
						mpfr_set_str( t2, "2.0", 10, MPFR_RNDN);
						mpfr_mul_ui(t2,t2,ii,MPFR_RNDU);
						mpfr_add_ui(t2,t2,1,MPFR_RNDD);
						mpfr_mul(t2,t2,eta,MPFR_RNDN);
						mpfr_mul(t2,t2,Pieta[ii],MPFR_RNDU);
						mpfr_sub(t2,t2,t1,MPFR_RNDD);
						mpfr_set_ui(t3, ii, MPFR_RNDN);
						mpfr_add_ui(t3,t3,1,MPFR_RNDD);
						mpfr_div(t2,t2,t3,MPFR_RNDD);
						mpfr_set(Pieta[ii+1], t2, MPFR_RNDD);
					}
					mpfr_set_str(GmultiP, "0.000",10, MPFR_RNDN);
					mpfr_t t11,t21,t31;
					mpfr_t t12,t22,t32;
					if(zijF<=rpF)
					{
						#pragma omp parallel shared(precd,GmultiP,Pimu,Pieta,zi,rp) private(t11,t21,t31)
						{
							mpfr_inits2 (precd,t11,t21,t31,(mpfr_ptr) 0);
							#pragma omp for private(n)
							for(n = 0; n <= 1000; n=n+2)
							{	
								mpfr_pow_ui(t11,rp,2*n+1,MPFR_RNDD);
								mpfr_ui_div(t11,1,t11,MPFR_RNDD);
								mpfr_ui_sub(t11,1,t11,MPFR_RNDD);
								mpfr_pow_ui(t21,zi,2,MPFR_RNDU);
								mpfr_mul(t11,t11,t21,MPFR_RNDD);
								mpfr_mul(t31,rp,zi,MPFR_RNDD);
								mpfr_pow_ui(t31,t31,n,MPFR_RNDU);
								mpfr_mul(t11,t11,t31,MPFR_RNDD);
								mpfr_mul(t11,t11,Pimu[n],MPFR_RNDD);
								mpfr_mul(t11,t11,Pieta[n],MPFR_RNDD);
								mpfr_div_ui(t11,t11,2*n+1,MPFR_RNDD);
								#pragma omp critical
								mpfr_add(GmultiP,GmultiP,t11,MPFR_RNDD);
							}
								mpfr_clear (t11);
								mpfr_clear (t21);
								mpfr_clear (t31);
						}	
					}
					if(zijF>rpF)
					{
						#pragma omp parallel shared(precd,GmultiP,Pimu,Pieta,zi,rp) private(t12,t22,t32)
						{
							mpfr_inits2 (precd,t12,t22,t32,(mpfr_ptr) 0);
							#pragma omp for private(n)
							for(n = 0; n <= 1000; n=n+2)
							{	
								mpfr_pow_ui(t12,zi,2*n+1,MPFR_RNDD);
								mpfr_ui_div(t12,1,t12,MPFR_RNDD);
								mpfr_ui_sub(t12,1,t12,MPFR_RNDD);
								mpfr_pow_ui(t22,zi,2,MPFR_RNDU);
								mpfr_mul(t12,t12,t22,MPFR_RNDD);
								mpfr_mul(t32,rp,zi,MPFR_RNDD);
								mpfr_pow_ui(t32,t32,n,MPFR_RNDU);
								mpfr_mul(t12,t12,t32,MPFR_RNDD);
								mpfr_mul(t12,t12,Pimu[n],MPFR_RNDD);
								mpfr_mul(t12,t12,Pieta[n],MPFR_RNDD);
								mpfr_div_ui(t12,t12,2*n+1,MPFR_RNDD);
								#pragma omp critical
								mpfr_add(GmultiP,GmultiP,t12,MPFR_RNDD);
							}
							mpfr_clear (t12);
							mpfr_clear (t22);
							mpfr_clear (t32);
						}	
					}
					mpfr_fprintf(fp,"%0.30Rg \n",GmultiP);
				}
			}
		}
	}
	mpfr_clear (t1);
	mpfr_clear (t2);
	mpfr_clear (t3);
	mpfr_clear (zi);
	mpfr_clear (eta);
	mpfr_clear (rp);
	mpfr_clear (muq);		
	for(int nn = 0; nn <= 1099; nn=nn+1)
	{	
		mpfr_clear(Pimu[nn]);
		mpfr_clear(Pieta[nn]);
	}
	mpfr_clear (GmultiP);
	mpfr_free_cache ();				
	fclose(fp);
    	return 0;
}