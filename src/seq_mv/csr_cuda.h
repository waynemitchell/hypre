typedef struct cuda_CSRMatrix{
  HYPRE_Int     *i;
  HYPRE_Int     *j;
  
  HYPRE_Complex  *data;
  
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  HYPRE_Int copied;
  HYPRE_Real *l1_norms_device;
} cuda_CSRMatrix;

typedef struct cuda_Vector{
  
  HYPRE_Complex  *data;
  HYPRE_Int offset1,offset2;
  //HYPRE_Int registered;
  HYPRE_Int send_to_device;
} cuda_Vector;
