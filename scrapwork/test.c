#include <stdio.h>
//~ #include <lapack.h>

extern void dpotrf_(char *UPLO,
                   int *N,
                   double *A,
                   int *LDA,
                   int *INFO);

int main(){
    char uplo[1];
    int i = 0;
    int N = 3, LDA = 3, INFO = 0;
    double A[9] = {4, 12, -16, 12, 37, -43, -16, -43, 98};

    uplo[0] = 'u';

    for(i=0; i<3; i++){
        printf("%g  %g  %g\n", A[3*i], A[3*i+1], A[3*i+2]);
    }
    printf("\n");

    dpotrf_(uplo, &N, A, &LDA, &INFO);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (j>i)
                A[i*N + j] = 0.0;


    for(i=0; i<3; i++){
        printf("%g  %g  %g\n", A[3*i], A[3*i+1], A[3*i+2]);
    }

    return 0;
}
