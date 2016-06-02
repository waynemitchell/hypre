#include <stdio.h>

#define gpuErrchk2(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
inline void gpuAssert2(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
extern "C"{
__global__
void VecScaleKernel(double *u, double *v, double *l1_norm, int num_rows){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i<10) printf("%d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  if (i<num_rows){
    u[i]+=v[i]/l1_norm[i];
    if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
  }
}
}

extern "C"{
  void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    int num_blocks=num_rows/32+1;
    //printf("Vecscale in Kernale call %d %d = %d %d\n",num_blocks,num_rows,num_blocks*32,sizeof(int));
    //printf("ARG Pointers %p %p %p\n",u,v,l1_norm);
    gpuErrchk2(cudaPeekAtLastError());
    gpuErrchk2(cudaDeviceSynchronize());
    VecScaleKernel<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    //dummy<<<num_blocks,32,0,s>>>(u,v,l1_norm,num_rows);
    gpuErrchk2(cudaPeekAtLastError());
    gpuErrchk2(cudaDeviceSynchronize());
  }
}

extern "C"{
  void VecScaleGSL(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s){
    //int num_blocks=num_rows/32+1;
    //printf("Vecscale %d %d = %d \n",num_blocks,num_rows,num_blocks*32);
    VecScaleKernelGSL<<<1024,32,0,s>>>(u,v,l1_norm,num_rows);
  }
}
