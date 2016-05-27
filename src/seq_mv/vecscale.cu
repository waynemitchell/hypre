__global__
void VecScaleKernel(double *__restrict__ u, double* __restrict__ v, double* __restrict__ l1_norm, int num_rows){
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < num_rows; 	 
       i += blockDim.x * gridDim.x) {
    u[i]+=v[i]/l1_norm[i];
  }
}
extern "C"{
void VecScale(double *u, double *v, double *l1_norm, int num_rows){
  
  VecScaleKernel<<<1024,32>>>(u,v,l1_norm,num_rows);
}
}
