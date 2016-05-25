typedef struct cuda_CSRMatrix{
  HYPRE_Int     *i;
  HYPRE_Int     *j;
  
  HYPRE_Complex  *data;
  
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  HYPRE_Int copied;
} cuda_CSRMatrix;

typedef struct cuda_Vector{
  
  HYPRE_Complex  *data;
  
} cuda_Vector;
