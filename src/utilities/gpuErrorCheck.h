/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/
#include <signal.h>
#if defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_SMS)
#include <cuda_runtime_api.h>
#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
#define MEM_PAD_LEN 1
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA ERROR ( Code = %d) in line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUDA ERROR : %s \n", cudaGetErrorString(code));
     raise(SIGABRT);
     exit(2);
   }
}

typedef enum memoryType {memoryTypeHost = 0, 
                 memoryTypeDevice = 1, 
                 memoryTypeHostPinned = 2, 
			 memoryTypeManaged = 3} memoryType_t;
/* typedef enum memoryType memoryType_t; */
inline memoryType_t queryPointer(const void *ptr)
{
  CUpointer_attribute attr[] = {CU_POINTER_ATTRIBUTE_CONTEXT,
                                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                CU_POINTER_ATTRIBUTE_IS_MANAGED};
  CUcontext context = NULL;
  CUmemorytype mem_type;
  int is_managed = 0;
  void* data[] = {&context, &mem_type, &is_managed};

  CUresult err = cuPointerGetAttributes(3, attr, data, (CUdeviceptr)ptr);
  if (err != CUDA_SUCCESS) {
    printf("queryPointer error: %d\n", err);
    exit(1);
  }

  memoryType_t type;

  if (context == NULL) {
    type = memoryTypeHost;
  } else {
    if (mem_type == CU_MEMORYTYPE_DEVICE) {
      if (is_managed)
        type = memoryTypeManaged;
      else
        type = memoryTypeDevice;
    } else {
      type = memoryTypeHostPinned;
    }
  }

  return type;
}

#define HYPRE_HOST_POINTER 0
#define HYPRE_MANAGED_POINTER 1
#define HYPRE_PINNED_POINTER 2
#define HYPRE_DEVICE_POINTER 3
#define HYPRE_UNDEFINED_POINTER1 4
#define HYPRE_UNDEFINED_POINTER2 5
void cudaSafeFree(void *ptr,int padding);
hypre_int PrintPointerAttributes(const void *ptr);
hypre_int PointerAttributes(const void *ptr);
#endif

#if defined(HYPRE_USE_GPU) && (defined(HYPRE_USE_MANAGED) || defined(HYPRE_USE_SMS))
#ifndef __cusparseErrorCheck__
#define __cusparseErrorCheck__
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
//#include <cuda_runtime_api.h>
#include <stdlib.h>
static inline const char *cusparseErrorCheck(cusparseStatus_t error)
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
        default:
	    return "Unknown error in cusparseErrorCheck";
    }
    
}
static inline const char *cublasErrorCheck(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
	    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
	    return "Unknown error in cublasErrorCheck";
    }

}
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char *file, int line)
//{
//   if (code != cudaSuccess) 
//   {
//     fprintf(stderr,"CUDA ERROR ( Code = %d) in line %d of file %s\n",code,line,file);
//     fprintf(stderr,"CUDA ERROR : %s \n", cudaGetErrorString(code));
//     exit(2);
//   }
//}
#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
static inline void cusparseAssert(cusparseStatus_t code, const char *file, int line)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
     fprintf(stderr,"CUSPARSE ERROR  ( Code = %d) IN CUDA CALL line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUSPARSE ERROR : %s \n", cusparseErrorCheck(code));
   }
}
#define cublasErrchk(ans){ cublasAssert((ans), __FILE__, __LINE__); }
static inline void cublasAssert(cublasStatus_t code, const char *file, int line)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
     fprintf(stderr,"CUBLAS ERROR  ( Code = %d) IN CUDA CALL line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUBLAS ERROR : %s \n", cublasErrorCheck(code));
   }
}
//int PointerType(const void *ptr);
void cudaSafeFree(void *ptr,int padding);
//void PrintPointerAttributes(const void *ptr);
//size_t mempush(void* ptr, size_t size,int purge);
//int memloc(void *ptr, int device);
#endif
#endif

