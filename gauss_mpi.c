#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

void prt1a(char *t1, double *v, int n,char *t2);
void wtime(double *t) {
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, (void *)0);
	if (sec < 0) sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

int N;
double *A;
#define A(i,j) A[(i)*(N+1)+(j)]
double *X;
int *map;

int main(int argc,char **argv) {
	double time0, time1;
	int rank, nprocs;
	int i, j, k;
	/* create arrays */
	MPI_Init(&argc, &argv);
	for (N=100; N < 2000; N += 200) {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		A=(double *)malloc(N*(N+1)*sizeof(double));
		X=(double *)malloc(N*sizeof(double));
		map=(int *)malloc(N*sizeof(int));
		if (rank == 0) {
			printf("GAUSS %dx%d\n----------------------------------\n",N,N);
			/* initialize array A*/
			for(i=0; i <= N-1; i++)
				for(j=0; j <= N; j++)
					/* this matrix i use for debug */
					if (j==N) {
						A(i,j) = N;
					} else if (i >= j) {
						A(i,j) = i + 1 + j;
					} else {
						A(i,j) = 0;
					}
			wtime(&time0);
		}
		MPI_Bcast (A,N*(N+1),MPI_DOUBLE,0,MPI_COMM_WORLD);
		for (i=0; i<N; i++) {
			map[i] = i % nprocs;
		} 
		/* elimination */
		for (i=0; i<=N-1; i++) {
			MPI_Bcast (&A(i,i),N-i+1,MPI_DOUBLE,map[i],MPI_COMM_WORLD);
			for (k=i+1; k <= N-1; k++) {
				if (map[k] == rank) {
					for (j=i+1; j <= N; j++) 
						A(k,j) -= A(k,i)*A(i,j)/A(i,i);
				}
			}
		}
		if (rank == 0) {
		/* reverse substitution */
			X[N-1] = A(N-1,N)/A(N-1,N-1);
			for (j=N-2; j>=0; j--) {
				for (k=0; k <= j; k++)
					A(k,N) = A(k,N)-A(k,j+1)*X[j+1];
				X[j]=A(j,N)/A(j,j);
			}
			wtime(&time1);
			printf("Time in seconds=%gs\n",time1-time0);
			prt1a("X=(", X,N>100?100:N,"...)\n");
		}
		free(A);
		free(X);
		free(map);
	}
	if (rank == 0) {
		printf("\n");
	}
	MPI_Finalize();
	return 0;
}

void prt1a(char * t1, double *v, int n,char *t2) {
	int j;
	printf("%s",t1);
	for(j=0;j<n;j++)
		printf("%.4g%s",v[j], j%10==9? "\n": ", ");
	printf("%s",t2);
}
