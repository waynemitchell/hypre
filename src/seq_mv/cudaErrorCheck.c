#include "cudaErrorCheck.h"
extern const char *cusparseErrorCheck(cusparseStatus_t error);
extern void gpuAssert(cudaError_t code, const char *file, int line);
extern void cusparseAssert(cusparseStatus_t code, const char *file, int line);
