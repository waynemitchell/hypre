#ifndef __cusparseErrorCheck__
#define __cusparseErrorCheck__
#include <stdio.h>
#include <cuda_runtime_api.h>
extern inline const char *cusparseErrorCheck(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "Congrats::Undefined ERRROR";
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
extern inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA ERROR ( Code = %d) in line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUDA ERROR : %s \n", cudaGetErrorString(code));
   }
}
#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
extern inline void cusparseAssert(cusparseStatus_t code, const char *file, int line)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
     fprintf(stderr,"CUSPARSE ERROR  ( Code = %d) IN CUDA CALL line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUSPARSE ERROR : %s \n", cusparseErrorCheck(code));
   }
}
int PointerType(const void *ptr);
#endif
