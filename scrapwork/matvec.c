#include <stdio.h>
//~ #include <lapack.h>

extern void dpotrf_(char *UPLO,
                   int *N,
                   double *A,
                   int *LDA,
                   int *INFO);

extern void dgemv_(char *TRANS,
                   int *M,
                   int *N,
                   double *ALPHA,
                   double *A,
                   int *LDA,
                   double *X,
                   int *INCX,
                   double *BETA,
                   double *Y,
                   int *INCY);

int main(){
    char uplo[1];
    int i = 0, j = 0;
    int N = 3, LDA = 3, INFO = 0;
    double A[9] = {4, 12, -16, 12, 37, -43, -16, -43, 98};

    char trans[1];
    int stride = 1;
    double one = 1.0;
    double zero = 0.0;
    double x[3] = {0, 0, 1};
    double y[3] = {0, 0, 0};

    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            printf("%g\t", A[N*i+j]);
        }
        printf("\n");
    }
    printf("\n");

    uplo[0] = 'u';
    dpotrf_(uplo, &N, A, &LDA, &INFO);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j>i)
                A[i*N + j] = 0.0;


    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            printf("%g\t", A[N*i+j]);
        }
        printf("\n");
    }
    printf("\n");


    for(i=0; i<N; i++){
        printf("%g\n", x[i]);
    }
    printf("\n");

    trans[0] = 'T';
    dgemv_(trans, &N, &N, &one, A, &LDA, x, &stride, &zero, y, &stride);

    for(i=0; i<N; i++){
        printf("%g\n", y[i]);
    }
    printf("\n");

    return 0;
}
