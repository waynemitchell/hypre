#include <stdio.h>
__device__ double norm_agg;
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

#define gpuErrchk2(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
     exit(2);
   }
}


__global__
void VecScaleKernelGSL(double *__restrict__ u, double* __restrict__ v, double* __restrict__ l1_norm, int num_rows){
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 	 
       i += blockDim.x * gridDim.x) {
    u[i]+=v[i]/l1_norm[i];
  }

}
__global__
void dummy(double *a, double *b,double *c, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<10) printf("Hello world %d %lf %lf %lf\n",num_rows,a[0],b[0],c[0]);
}

__global__
void PrintDeviceArrayKernel(double *a,int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<num_rows) printf("PrintARRAYDEVICE %d %lf\n",i,a[i]);
}

__global__
void PrintVecNormKernel(double *a,int tag1, int tag2, double comp,int num_rows){
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double ssum[32];
  ssum[ii]=0.0;
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 	 
       i += blockDim.x * gridDim.x) {
    ssum[threadIdx.x]+=a[i];
  }
  __syncthreads();
  if (ii==0) {
    double norm=0.0;
    int j;
    for(j=0;j<32;j++) norm+=ssum[j];
    printf("VECNORM %d IS %lf == %lf %d\n",tag1*1000+tag2,norm,comp,num_rows);
  }
}
  
extern "C"{
__global__
void VecScaleKernel(double *__restrict__ u, double *__restrict__ v, double *__restrict__ l1_norm, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    u[i]+=v[i]/l1_norm[i];
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}

extern "C"{
  void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    int num_blocks=num_rows/32+1;
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    VecScaleKernel<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
  }
}

extern "C"{
  void VecScaleGSL(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    //int num_blocks=num_rows/32+1;
    //printf("Vecscale %d %d = %d \n",num_blocks,num_rows,num_blocks*32);
    VecScaleKernelGSL<<<1024,32,0,s>>>(u,v,l1_norm,num_rows);
  }
}

extern "C"{
  void PrintDeviceVec(double *u, int num_rows,cudaStream_t s){
    int num_blocks=num_rows/32+1;
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARG Pointers %p %p %p\n",u,v,l1_norm);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    PrintDeviceArrayKernel<<<num_blocks,32,0,s>>>(u,num_rows);
  }
}

extern "C"{
  void PrintVecNorm(double *u, int num_rows,int tag1, int tag2, double comp,cudaStream_t s)
  {
    //int num_blocks=num_rows/32+1;
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARGs %p %d %d\n",u,tag1,tag2);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    PrintVecNormKernel<<<1,32,0,s>>>(u,tag1,tag2, comp,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
  }
}

// Mods that calculate the l1_norm locally

extern "C"{
__global__
void VecScaleKernelWithNorms1(double *__restrict__ u, double *__restrict__ v, double *__restrict__ l1_norm, 
			     int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			     int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double ll1_norm=0.0;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    int j;
    for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
      ll1_norm += fabs(A_diag_data[j]);
    for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
      ll1_norm += fabs(A_offd_data[j]);
    u[i]+=v[i]/ll1_norm;
    l1_norm[i]=ll1_norm;
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
__global__
void VecScaleKernelWithNorms2(double *__restrict__ u, double *__restrict__ v, double *__restrict__ l1_norm, 
			     int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			     int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double ll1_norm=0.0;
  //if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    int j;
    for (j = A_diag_I[i]; j < A_diag_I[i+1]; j++)
      ll1_norm += fabs(A_diag_data[j]);
    //for (j = A_offd_I[i]; j < A_offd_I[i+1]; j++)
    //  l1_norm += fabs(A_offd_data[j]);
    u[i]+=v[i]/ll1_norm;
    l1_norm[i]=ll1_norm;
    //if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}
extern "C"{
  void VecScaleWithNorms(double *u, double *v, double *l1_norm, 
			 int *A_diag_I,  double *A_diag_data, int *A_offd_I,double *A_offd_data,
			 int num_rows,cudaStream_t s){
    int tpb=64;
    int num_blocks=num_rows/tpb+1;
    // Let cuda caluclate gridsize and block size
    if (0){
    int minGridSize,blockSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
					VecScaleKernelWithNorms1, 0, 0); 
    printf("Grid & Block size for max occupancy Norm1 %d(%d) %d(%d)\n",minGridSize,num_blocks,blockSize,tpb);
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      VecScaleKernelWithNorms2, 0, 0); 
    printf("Grid & Block size for max occupancy Norm2 %d(%d) %d(%d)\n",minGridSize,num_blocks,blockSize,tpb);
    }
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARG Pointers %p %p %p\n",u,v,l1_norm);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
    if (A_offd_I)
      VecScaleKernelWithNorms1<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,A_diag_I,A_diag_data,A_offd_I,A_offd_data,num_rows);
  else
    VecScaleKernelWithNorms2<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,A_diag_I,A_diag_data,A_offd_I,A_offd_data,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //gpuErrchk2(cudaPeekAtLastError());
    //gpuErrchk2(cudaDeviceSynchronize());
  }
}
extern "C"{
  void PrintVecNormHost(double *u, int num_rows,int tag1, int tag2, double comp,cudaStream_t s){
    double norm=0.0;
    for(int i=0;i<num_rows;i++) norm+=u[i];
    printf("VECNORM %d IS %lf == %lf \n",tag1*1000+tag2,norm,comp);
  }
}
