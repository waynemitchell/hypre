// Helper functions to setup amgdd composite grids

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"   

#ifdef __cplusplus

#include <vector>
#include <map>
#include <set>

// !!! Timing
#include <chrono>
#include <iostream>

#if defined(HYPRE_USING_GPU)
#include "cuda_profiler_api.h"
#include <thrust/transform_scan.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#endif

using namespace std;

extern "C"
{

#endif

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost );

HYPRE_Int
UnpackRecvBuffer( HYPRE_Int *recv_buffer, hypre_ParCompGrid **compGrid, 
      hypre_ParCSRCommPkg *commPkg,
      HYPRE_Int **A_tmp_info,
      hypre_ParCompGridCommPkg *compGridCommPkg,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ****recv_redundant_marker, HYPRE_Int ***num_recv_nodes, 
      HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels,
      HYPRE_Int *nodes_added_on_level, HYPRE_Int buffer_number, HYPRE_Int *num_resizes, 
      HYPRE_Int symmetric);

HYPRE_Int* PackSendBuffer(hypre_ParAMGData *amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, HYPRE_Int *send_flag_buffer_size, 
   HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes, HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, 
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric, HYPRE_Real *total_timings );

HYPRE_Int LocalToGlobalIndex(hypre_ParCompGrid *compGrid, HYPRE_Int local_index);

#ifdef __cplusplus
}



#if defined(HYPRE_USING_GPU)
extern "C"
{
   __global__
   void ExpandPaddingNewKernel(HYPRE_Int *add_flag,
                        HYPRE_Int *owned_col_ind,
                        HYPRE_Int *nonowned_col_ind,
                        HYPRE_Int *owned_row_ptr,
                        HYPRE_Int *nonowned_row_ptr,
                        HYPRE_Int add_flag_size,
                        HYPRE_Int add_flag_start,
                        HYPRE_Int num_owned,
                        HYPRE_Int dist)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      HYPRE_Int j, jj;
      if (i < add_flag_size)
      {
         if (add_flag[i + add_flag_start] == dist)
         {
            HYPRE_Int start = owned_row_ptr[i];
            HYPRE_Int end = owned_row_ptr[i+1];
            for (jj = start; jj < end; jj++)
            {
               j = owned_col_ind[jj];
               add_flag[j] = max(add_flag[j], dist-1);
            }
            start = nonowned_row_ptr[i];
            end = nonowned_row_ptr[i+1];
            for (jj = start; jj < end; jj++)
            {
               j = nonowned_col_ind[jj];
               if (j >= 0) add_flag[j + num_owned] = max(add_flag[j + num_owned], dist-1);
            }
         }     
      } 
   }

   __global__
   void PackColIndNewKernel(HYPRE_Int *local_indices, 
                        HYPRE_Int *destination,
                        HYPRE_Int *offsets,
                        HYPRE_Int *owned_diag_col_ind,
                        HYPRE_Int *owned_offd_col_ind,
                        HYPRE_Int *nonowned_diag_col_ind,
                        HYPRE_Int *nonowned_offd_col_ind,
                        HYPRE_Int *owned_diag_row_ptr,
                        HYPRE_Int *owned_offd_row_ptr,
                        HYPRE_Int *nonowned_diag_row_ptr,
                        HYPRE_Int *nonowned_offd_row_ptr,
                        HYPRE_Int *nonowned_global_ind,
                        HYPRE_Int *add_flag,
                        HYPRE_Int local_indices_size,
                        HYPRE_Int num_owned,
                        HYPRE_Int first_global_index)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      HYPRE_Int j;
      if (i < local_indices_size)
      {
         HYPRE_Int cnt = offsets[i];
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt+1);

         if (send_elmt < num_owned)
         {
            HYPRE_Int start = owned_diag_row_ptr[send_elmt];
            HYPRE_Int end = owned_diag_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = owned_diag_col_ind[j];
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(add_flag_index + first_global_index + 1); // -(GID + 1)
               }
            }
            start = owned_offd_row_ptr[send_elmt];
            end = owned_offd_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = owned_offd_col_ind[j] + num_owned;
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(nonowned_global_ind[ owned_offd_col_ind[j] ] + 1); // -(GID + 1)
               }
            }
         }
         else
         {
            send_elmt -= num_owned;
            HYPRE_Int start = nonowned_diag_row_ptr[send_elmt];
            HYPRE_Int end = nonowned_diag_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               if (nonowned_diag_col_ind[j] >= 0)
               {
                  HYPRE_Int add_flag_index = nonowned_diag_col_ind[j] + num_owned;
                  if (add_flag[add_flag_index] > 0)
                  {
                     destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
                  }
                  else
                  {
                     destination[cnt++] = -(nonowned_global_ind[ nonowned_diag_col_ind[j] ] + 1); // -(GID + 1)
                  }
               }
               else
               {
                  destination[cnt++] = nonowned_diag_col_ind[j]; // -(GID + 1)
               }
            }
            start = nonowned_offd_row_ptr[send_elmt];
            end = nonowned_offd_row_ptr[send_elmt+1];
            for (j = start; j < end; j++)
            {
               HYPRE_Int add_flag_index = nonowned_offd_col_ind[j];
               if (add_flag[add_flag_index] > 0)
               {
                  destination[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  destination[cnt++] = -(add_flag_index + first_global_index + 1); // -(GID + 1)
               }
            }
         }
      } 
   }

   __global__
   void AddFlagToSendFlagKernel(HYPRE_Int *marker, HYPRE_Int *aux_marker, HYPRE_Int *list, HYPRE_Int marker_size, HYPRE_Int ghost_dist)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < marker_size)
      {
          if (marker[i] > ghost_dist) list[ aux_marker[i] ] = i;
          else if (marker[i] > 0) list[ aux_marker[i] ] = -(i + 1);
      }
   }

   struct greater_than_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x > 0;
      }
   };

   struct less_than_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x < 0;
      }
   };

   struct greater_than_equal_zero
   {
      __host__ __device__
      HYPRE_Int operator()(HYPRE_Int x)
      {
         return x >= 0;
      }
   };

   __global__
   void GetGlobalIndexKernel(HYPRE_Int *lid, HYPRE_Int *gid, HYPRE_Int *nonowned_gid, HYPRE_Int list_size, HYPRE_Int owned_size, HYPRE_Int first_owned_gid)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int local_index = lid[i];
         if (local_index < 0) local_index = -(local_index+1);
         if (local_index < owned_size) gid[i] = local_index + first_owned_gid;
         else gid[i] = nonowned_gid[local_index - owned_size];
      }
   }

   void SortSendFlagGPU(hypre_ParCompGrid *compGrid, HYPRE_Int *list, HYPRE_Int list_size)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int num_blocks = list_size/tpb+1;

      // Setup the keys as global indices
      HYPRE_Int *gid = hypre_CTAlloc(HYPRE_Int, list_size, HYPRE_MEMORY_DEVICE);
      GetGlobalIndexKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(list, gid, hypre_ParCompGridNonOwnedGlobalIndices(compGrid), list_size, hypre_ParCompGridNumOwnedNodes(compGrid), hypre_ParCompGridFirstGlobalIndex(compGrid));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      
      // Sort the list by global index
      thrust::sort_by_key(thrust::cuda::par.on(HYPRE_STREAM(1)), gid, gid + list_size, list);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      hypre_TFree(gid, HYPRE_MEMORY_DEVICE);
   }

   HYPRE_Int* AddFlagToSendFlagGPU(hypre_ParCompGrid *compGrid,
                        HYPRE_Int *add_flag,
                        HYPRE_Int *num_send_nodes,
                        HYPRE_Int num_ghost_layers)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid);
      HYPRE_Int num_blocks = total_num_nodes/tpb+1;
      
      // Generate array that contains gather locations 
      HYPRE_Int *gather_locations = hypre_CTAlloc(HYPRE_Int, total_num_nodes, HYPRE_MEMORY_DEVICE);
      thrust::transform_exclusive_scan(thrust::cuda::par.on(HYPRE_STREAM(1)), 
                                          add_flag, 
                                          add_flag + total_num_nodes, 
                                          gather_locations, 
                                          greater_than_zero(),  
                                          0, 
                                          thrust::plus<HYPRE_Int>());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // Allocate send_flag
      if (add_flag[total_num_nodes-1] == 0) (*num_send_nodes) = gather_locations[total_num_nodes-1];
      else (*num_send_nodes) = gather_locations[total_num_nodes-1] + 1;
      HYPRE_Int *send_flag = hypre_CTAlloc(HYPRE_Int, (*num_send_nodes), HYPRE_MEMORY_DEVICE);

      // Collapse from marker to send_flag
      AddFlagToSendFlagKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(add_flag, gather_locations, send_flag, total_num_nodes, num_ghost_layers);
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      hypre_TFree(gather_locations, HYPRE_MEMORY_DEVICE);
      
      SortSendFlagGPU(compGrid, send_flag,(*num_send_nodes));
     
      return send_flag;
   }

   __global__
   void ListToMarkerKernel(HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int list_size)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
          HYPRE_Int marker_idx = list[i];
          if (marker_idx < 0) marker_idx = -(marker_idx + 1);
          marker[ marker_idx ] = i + 1;
      }
   }

   __global__
   void GetCoarseIndicesKernel(HYPRE_Int *list,
              HYPRE_Int *coarse_list,
              HYPRE_Int *owned_coarse_indices,
              HYPRE_Int *nonowned_coarse_indices,
              HYPRE_Int num_owned,
              HYPRE_Int num_owned_coarse,
              HYPRE_Int list_size)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int idx = list[i];
         if (idx < num_owned)
         {
            coarse_list[i] = owned_coarse_indices[idx];
         }
         else
         {
            idx -= num_owned;
            HYPRE_Int coarse_index = nonowned_coarse_indices[idx];
            if (coarse_index >= 0)
               coarse_list[i] = coarse_index + num_owned_coarse;
            else
               coarse_list[i] = coarse_index;
         }
      }
   }

   struct not_in_range
   {
      HYPRE_Int lower;
      HYPRE_Int upper;
      __host__ __device__
      bool operator()(const int x)
      {
         return (x < lower || x >= upper);
      }
   };
   struct minus_equal
   {
      HYPRE_Int a;
      __host__ __device__
      HYPRE_Int operator()(const HYPRE_Int x)
      {
         return x - a;
      }
   };

   void MarkCoarseGPU(hypre_ParCompGrid *compGrid,
                        HYPRE_Int *send_flag,
                        HYPRE_Int num_send_nodes,
                        HYPRE_Int *add_flag, 
                        HYPRE_Int num_owned_coarse,
                        HYPRE_Int dist)
   {
      // Owned points 
      not_in_range owned_range;
      owned_range.lower = 0;
      owned_range.upper = hypre_ParCompGridNumOwnedNodes(compGrid);
      thrust::device_vector<HYPRE_Int> owned_real_indices_d(num_send_nodes); 
      thrust::device_vector<HYPRE_Int> owned_coarse_indices_d(num_send_nodes); 

      // !!! Debug
      HYPRE_Int myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
      if (myid == 1)
      {
         HYPRE_Int i;
         printf("send flag = ");
         for (i = 0; i < num_send_nodes; i++)
            printf("%d ", send_flag[i]);
         printf("\n");
         printf("sequential coarse indices = ");
         for (i = 0; i < num_send_nodes; i++)
            if (send_flag[i] >= 0)
               printf("%d ", hypre_ParCompGridOwnedCoarseIndices(compGrid)[ send_flag[i] ]);
         printf("\n");
      }

      auto owned_real_indices_end = thrust::remove_copy_if(thrust::cuda::par.on(HYPRE_STREAM(1)),
                                                         send_flag,
                                                         send_flag + num_send_nodes,
                                                         send_flag,
                                                         owned_real_indices_d.begin(),
                                                         owned_range); 	
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
      HYPRE_Int num_owned_real = owned_real_indices_end - owned_real_indices_d.begin();

      // !!! Debug
      /* if (myid == 1) */
      /* { */
      /*    thrust::host_vector<HYPRE_Int> owned_real_indices_h = owned_real_indices_d; */
      /*    HYPRE_Int i; */
      /*    printf("owned real indices = "); */
      /*    for (auto it = owned_real_indices_h.begin(); it != owned_real_indices_h.end(); ++it) */
      /*       printf("%d, ", *it); */
      /*    printf("\n"); */
      /* } */

      thrust::gather(thrust::cuda::par.on(HYPRE_STREAM(1)),
                        owned_real_indices_d.begin(),
                        owned_real_indices_end,
                        hypre_ParCompGridOwnedCoarseIndices(compGrid),
                        owned_coarse_indices_d.begin());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // !!! Debug
      if (myid == 1)
      {
         thrust::host_vector<HYPRE_Int> owned_coarse_indices_h = owned_coarse_indices_d;
         HYPRE_Int i;
         printf("    owned coarse indices = ");
         for (auto it = owned_coarse_indices_h.begin(); it != owned_coarse_indices_h.end(); ++it)
            printf("%d ", *it);
         printf("\n");
      }

      thrust::constant_iterator<HYPRE_Int> dist_it(dist);
      thrust::scatter_if(thrust::cuda::par.on(HYPRE_STREAM(1)),
                           dist_it,
                           dist_it + (owned_real_indices_end - owned_real_indices_d.begin()),
                           owned_coarse_indices_d.begin(),
                           owned_coarse_indices_d.begin(),
                           add_flag,
                           greater_than_equal_zero());
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));

      // !!! Debug
      if (myid == 1)
      {
         HYPRE_Int i;
         printf("add_flag = ");
         for (i = 0; i < num_owned_coarse; i++)
            printf("%d, ", add_flag[i]);
         printf("\n");
      }
      // Nonowned points 
      /* not_in_range nonowned_range; */
      /* nonowned_range.lower = 0; */
      /* nonowned_range.upper = hypre_ParCompGridNumNonOwnedNodes(compGrid); */
      /* minus_equal minus_num_owned; */
      /* minus_num_owned.a = hypre_ParCompGridNumOwnedNodes(compGrid); */
      /* thrust::device_vector<HYPRE_Int> nonowned_scatter_indices_d(num_send_nodes); */ 
      /* thrust::gather_if(thrust::cuda::par.on(HYPRE_STREAM(2)), */
      /*                      thrust::make_transform_iterator(send_flag, minus_num_owned), */
      /*                      thrust::make_transform_iterator(send_flag + num_send_nodes, minus_num_owned), */
      /*                      thrust::make_transform_iterator(send_flag, minus_num_owned), */
	                        /* hypre_ParCompGridNonOwnedCoarseIndices(compGrid), */
      /*                      nonowned_scatter_indices_d.begin(), */
      /*                      nonowned_range); */
      /* thrust::scatter(thrust::cuda::par.on(HYPRE_STREAM(2)), */
      /*                      dist_it, */
      /*                      dist_it + num_send_nodes, */
      /*                      nonowned_scatter_indices_d.begin(), */
      /*                      add_flag + num_owned_coarse); */
      hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(2))); */

   }

   __global__
   void MarkCoarseKernel(HYPRE_Int *list,
              HYPRE_Int *marker,
              HYPRE_Int *owned_coarse_indices,
              HYPRE_Int *nonowned_coarse_indices,
              HYPRE_Int num_owned,
              HYPRE_Int num_owned_coarse,
              HYPRE_Int list_size,
              HYPRE_Int dist,
              HYPRE_Int *nodes_to_add,
              HYPRE_Int myid)
   {

      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < list_size)
      {
         HYPRE_Int idx = list[i];
         if (idx >= 0)
         {
            if (idx < num_owned)
            {
               HYPRE_Int coarse_index = owned_coarse_indices[idx];
               if (coarse_index >= 0)
               {
                  marker[ coarse_index ] = dist;
                  (*nodes_to_add) = 1;
               }
            }
            else
            {
               idx -= num_owned;
               if (nonowned_coarse_indices[idx] >= 0)
               {
                  HYPRE_Int coarse_index = nonowned_coarse_indices[idx] + num_owned_coarse;
                  marker[ coarse_index ] = dist;
                  (*nodes_to_add) = 1;
               }
            }
         }
      }
   }

   __global__
   void LocalToGlobalIndexKernel(HYPRE_Int *local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index < 0) local_index = -(local_index + 1);

         if (local_index < num_owned)
            global_indices[i] = local_index + first_global;
         else
            global_indices[i] = nonowned_global_indices[local_index - num_owned];
      }
   }

   __global__
   void PackCoarseGlobalIndicesKernel(HYPRE_Int *local_indices, HYPRE_Int *destination, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global_coarse, HYPRE_Int *owned_coarse_indices, HYPRE_Int *nonowned_coarse_indices, HYPRE_Int *nonowned_global_indices_coarse)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt + 1);

         if (send_elmt < num_owned)
         {
            if (owned_coarse_indices[ send_elmt ] >= 0)
               destination[i] = owned_coarse_indices[ send_elmt ] + first_global_coarse;
            else
               destination[i] = owned_coarse_indices[ send_elmt ];
         }
         else 
         {
            HYPRE_Int nonowned_index = send_elmt - num_owned;
            HYPRE_Int nonowned_coarse_index = nonowned_coarse_indices[ nonowned_index ];
            
            if (nonowned_coarse_index >= 0)
            {
               destination[i] = nonowned_global_indices_coarse[ nonowned_coarse_index ];
            }
            else if (nonowned_coarse_index == -1)
               destination[i] = nonowned_coarse_index;
            else
               destination[i] = -(nonowned_coarse_index+2);
         }
      }
   }

   __global__
   void LocalToGlobalIndexMarkGhostKernel(HYPRE_Int *local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index < 0)
         {
            local_index = -(local_index + 1);
            if (local_index < num_owned)
               global_indices[i] = -(local_index + first_global + 1);
            else
               global_indices[i] = -(nonowned_global_indices[local_index - num_owned] + 1);
         }
         else
         {
            if (local_index < num_owned)
               global_indices[i] = local_index + first_global;
            else
               global_indices[i] = nonowned_global_indices[local_index - num_owned];
         }
      }
   }

   __global__
   void GetRowSizesKernel(HYPRE_Int *local_indices, HYPRE_Int *row_sizes, 
         HYPRE_Int *owned_diag_row_ptr,
         HYPRE_Int *owned_offd_row_ptr,
         HYPRE_Int *nonowned_diag_row_ptr,
         HYPRE_Int *nonowned_offd_row_ptr,
         HYPRE_Int local_indices_size, HYPRE_Int num_owned)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int send_elmt = local_indices[i];
         if (send_elmt < 0) send_elmt = -(send_elmt + 1);

         if (send_elmt < num_owned)
         {
            row_sizes[i] += owned_diag_row_ptr[send_elmt+1] - owned_diag_row_ptr[send_elmt];
            row_sizes[i] += owned_offd_row_ptr[send_elmt+1] - owned_offd_row_ptr[send_elmt];
         }
         else
         {
            send_elmt -= num_owned;
            row_sizes[i] += nonowned_diag_row_ptr[send_elmt+1] - nonowned_diag_row_ptr[send_elmt];
            row_sizes[i] += nonowned_offd_row_ptr[send_elmt+1] - nonowned_offd_row_ptr[send_elmt];
         }
      }
   }

   __global__
   void LocalToGlobalIndexRemoveGhostKernel(HYPRE_Int *local_indices, HYPRE_Int *aux_local_indices, HYPRE_Int *global_indices, HYPRE_Int local_indices_size, HYPRE_Int num_owned, HYPRE_Int first_global, HYPRE_Int *nonowned_global_indices)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < local_indices_size)
      {
         HYPRE_Int local_index = local_indices[i];
         if (local_index >= 0)
         {
            if (local_index < num_owned)
               global_indices[aux_local_indices[i]] = local_index + first_global;
            else
               global_indices[aux_local_indices[i]] = nonowned_global_indices[local_index - num_owned];
         }
      }
   }

   HYPRE_Int LocalToGlobalIndexRemoveGhostGPU(HYPRE_Int *local_indices, 
         HYPRE_Int **global_indices, 
         HYPRE_Int local_indices_size, 
         HYPRE_Int num_owned, 
         HYPRE_Int first_global, 
         HYPRE_Int *nonowned_global_indices)
   {
      const HYPRE_Int tpb=64;
      HYPRE_Int num_blocks = local_indices_size/tpb+1;
      
      HYPRE_Int *aux_local_indices = NULL;
      HYPRE_Int global_indices_size;
      // If not keeping ghost dofs, need to get locations to gather to and new size for global indices
      aux_local_indices = hypre_CTAlloc(HYPRE_Int, local_indices_size, HYPRE_MEMORY_DEVICE);
      greater_than_equal_zero gtez;
      thrust::transform(thrust::cuda::par.on(HYPRE_STREAM(1)), local_indices, local_indices + local_indices_size, aux_local_indices, gtez);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      thrust::exclusive_scan(thrust::cuda::par.on(HYPRE_STREAM(1)), aux_local_indices, aux_local_indices + local_indices_size, aux_local_indices);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */
      if (local_indices[local_indices_size-1] < 0) global_indices_size = aux_local_indices[local_indices_size-1];
      else global_indices_size = aux_local_indices[local_indices_size-1]+ 1;
      (*global_indices) = hypre_CTAlloc(HYPRE_Int, global_indices_size, HYPRE_MEMORY_DEVICE);
      LocalToGlobalIndexRemoveGhostKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(local_indices, aux_local_indices, (*global_indices), local_indices_size, num_owned, first_global, nonowned_global_indices);
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1))); */

      return global_indices_size;
   }

   void RemoveRedundancyGPU(hypre_ParAMGData *amg_data,
                              hypre_ParCompGrid **compGrid,
                              hypre_ParCompGridCommPkg *compGridCommPkg,
                              HYPRE_Int ****send_flag,
                              HYPRE_Int ****recv_map,
                              HYPRE_Int ***num_send_nodes,
                              HYPRE_Int current_level,
                              HYPRE_Int proc,
                              HYPRE_Int level,
                              cudaStream_t stream)
   {

      const HYPRE_Int tpb=64;

      HYPRE_Int   myid;
      hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
      // !!! Debug
      /* if (myid == 0) printf("In RemoveRedundancyGPU() %d, %d, %d\n", current_level, proc, level); */

      // Get the global indices for the send_flag
      HYPRE_Int *send_gid = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE);
      HYPRE_Int *keys_result = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE); 
      HYPRE_Int num_blocks = num_send_nodes[current_level][proc][level]/tpb+1;
      LocalToGlobalIndexKernel<<<num_blocks,tpb,0,stream>>>(send_flag[current_level][proc][level], send_gid, num_send_nodes[current_level][proc][level], hypre_ParCompGridNumOwnedNodes(compGrid[level]), hypre_ParCompGridFirstGlobalIndex(compGrid[level]), hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]));
      /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */

      // Will need a copy of the send flag
      HYPRE_Int *current_send_flag = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_DEVICE);
      
      // Eliminate redundant send info by comparing with previous send_flags and recv_maps
      HYPRE_Int current_send_proc = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[current_level][proc];
      HYPRE_Int prev_proc, prev_level;
      for (prev_level = current_level+1; prev_level <= level; prev_level++)
      {
         if (num_send_nodes[current_level][proc][level])
         {
            for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[prev_level]; prev_proc++)
            {
               if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc && num_send_nodes[prev_level][prev_proc][level])
               {
                  HYPRE_Int original_send_size = 0;
                  if (prev_level == level) 
                  {
                     hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[prev_level]);
                     HYPRE_Int original_proc;
                     for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumSends(original_commPkg); original_proc++)
                     {
                        if (hypre_ParCSRCommPkgSendProc(original_commPkg, original_proc) == current_send_proc) 
                        {
                           original_send_size = hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc);
                           break;
                        }
                     }
                  }

                  // Get the global indices (to be used as keys). NOTE: need to get size of prev send (we exclude ghosts when comparing), combine this with the info in original_send_size above.
                  cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  HYPRE_Int *prev_send_flag = send_flag[prev_level][prev_proc][level];

                  HYPRE_Int *prev_send_gid;
                  HYPRE_Int prev_send_gid_size = LocalToGlobalIndexRemoveGhostGPU(prev_send_flag, &prev_send_gid, num_send_nodes[prev_level][prev_proc][level], hypre_ParCompGridNumOwnedNodes(compGrid[level]), hypre_ParCompGridFirstGlobalIndex(compGrid[level]), hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]));
                  thrust::pair<HYPRE_Int*,HYPRE_Int*> end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_send_gid + original_send_size, prev_send_gid + prev_send_gid_size, current_send_flag, prev_send_flag, keys_result, send_flag[current_level][proc][level]);
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                  num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                  cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);

                  if (original_send_size)
                  {
                     cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                     end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_send_gid, prev_send_gid + original_send_size, current_send_flag, prev_send_flag, keys_result, send_flag[current_level][proc][level]);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                     cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  }
                  hypre_TFree(prev_send_gid, HYPRE_MEMORY_DEVICE);
               }
            }
         }

         if (num_send_nodes[current_level][proc][level])
         {
            for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[prev_level]; prev_proc++)
            {
               if (hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc && hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level])
               {
                  HYPRE_Int original_send_size = 0;
                  if (prev_level == level) 
                  {
                     hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[prev_level]);
                     HYPRE_Int original_proc;
                     for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumRecvs(original_commPkg); original_proc++)
                     {
                        if (hypre_ParCSRCommPkgRecvProc(original_commPkg, original_proc) == current_send_proc) 
                        {
                           original_send_size = hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc);
                           break;
                        }
                     }
                  }

                  // Get the global indices (to be used as keys). NOTE: need to get size of prev send (we exclude ghosts when comparing), combine this with the info in original_send_size above.
                  cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  HYPRE_Int *prev_recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level];
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */

                  HYPRE_Int *prev_recv_gid;
                  HYPRE_Int prev_recv_gid_size = LocalToGlobalIndexRemoveGhostGPU(prev_recv_map, &prev_recv_gid, hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level], hypre_ParCompGridNumOwnedNodes(compGrid[level]), hypre_ParCompGridFirstGlobalIndex(compGrid[level]), hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]));
                  thrust::pair<HYPRE_Int*,HYPRE_Int*> end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_recv_gid + original_send_size, prev_recv_gid + prev_recv_gid_size, current_send_flag, prev_recv_map, keys_result, send_flag[current_level][proc][level]);
                  /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                  num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                  cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);

                  if (original_send_size)
                  {
                     cudaMemcpyAsync(current_send_flag, send_flag[current_level][proc][level], sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     end = thrust::set_difference_by_key(thrust::cuda::par.on(stream), send_gid, send_gid + num_send_nodes[current_level][proc][level], 
                        prev_recv_gid, prev_recv_gid + original_send_size, current_send_flag, prev_recv_map, keys_result, send_flag[current_level][proc][level]);
                     /* hypre_CheckErrorDevice(cudaStreamSynchronize(stream)); */
                     num_send_nodes[current_level][proc][level] = end.second - send_flag[current_level][proc][level]; // NOTE: no realloc of send_flag here... do that later I guess when final size determined?
                     cudaMemcpyAsync(send_gid, keys_result, sizeof(HYPRE_Int)*num_send_nodes[current_level][proc][level], cudaMemcpyDeviceToDevice, stream);
                  }
                  hypre_TFree(prev_recv_gid, HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      hypre_TFree(current_send_flag, HYPRE_MEMORY_DEVICE);
      hypre_TFree(send_gid, HYPRE_MEMORY_DEVICE);
      hypre_TFree(keys_result, HYPRE_MEMORY_DEVICE);
      hypre_CheckErrorDevice(cudaStreamSynchronize(stream));
   }

}
#endif


HYPRE_Int
GetDofRecvProc(HYPRE_Int neighbor_local_index, hypre_ParCSRMatrix *A)
{
   HYPRE_Int *colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int *offdRowPtr = hypre_CSRMatrixI( hypre_ParCSRMatrixOffd(A) );

   // Use that column index to find which processor this dof is received from
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int recv_proc = -1;
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
   {
      if (neighbor_local_index >= hypre_ParCSRCommPkgRecvVecStart(commPkg,i) && neighbor_local_index < hypre_ParCSRCommPkgRecvVecStart(commPkg,i+1))
      {
         recv_proc = hypre_ParCSRCommPkgRecvProc(commPkg,i);
         break;
      }
   }

   return recv_proc;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int dof_index, HYPRE_Int distance, hypre_ParCSRMatrix *A, HYPRE_Int *add_flag, HYPRE_Int *add_flag_requests)
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i,j;

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);

   // Look at diag neighbors
   for (i = hypre_CSRMatrixI(diag)[dof_index]; i < hypre_CSRMatrixI(diag)[dof_index+1]; i++)
   {
      // Get the index of the neighbor
      HYPRE_Int neighbor_index = hypre_CSRMatrixJ(diag)[i];

      // If the neighbor info is available on this proc
      // And if we still need to visit this index (note that send_dofs[neighbor_index] = distance means we have already added all distance-1 neighbors of index)
      
      // See whether this dof is in the send dofs
      if (add_flag[neighbor_index] < distance)
      {
         add_flag[neighbor_index] = distance;
         if (distance - 1 > 0) RecursivelyFindNeighborNodes(neighbor_index, distance-1, A, add_flag, add_flag_requests);
      }
   }
   // Look at offd neighbors
   for (i = hypre_CSRMatrixI(offd)[dof_index]; i < hypre_CSRMatrixI(offd)[dof_index+1]; i++)
   {
      HYPRE_Int neighbor_index = hypre_CSRMatrixJ(offd)[i];

      if (add_flag_requests[neighbor_index] < distance)
         add_flag_requests[neighbor_index] = distance;
   }

   return 0;
}

HYPRE_Int
AddToSendAndRequestDofs(hypre_ParCSRMatrix *A, HYPRE_Int *add_flag, HYPRE_Int *add_flag_requests, 
   map<HYPRE_Int, HYPRE_Int> &send_dofs, 
   map< HYPRE_Int, map<HYPRE_Int, vector<pair<HYPRE_Int, HYPRE_Int> > > > &request_proc_dofs, HYPRE_Int destination_proc )
{
    for (auto i = 0; i < hypre_ParCSRMatrixNumRows(A); i++)
    {
        if (add_flag[i]) send_dofs[i] = add_flag[i]; // !!! Optimization: can probably not even use a map for the send dofs... avoid insert, just push to back of vector or something
    }
    for (auto i = 0; i < hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)); i++)
    {
        if (add_flag_requests[i])
        {
            HYPRE_Int neighbor_global_index = hypre_ParCSRMatrixColMapOffd(A)[i];
      
            HYPRE_Int recv_proc = GetDofRecvProc(i, A);
      
            if (recv_proc != destination_proc)
            {
                request_proc_dofs[recv_proc][destination_proc].push_back(pair<HYPRE_Int,HYPRE_Int>(neighbor_global_index,add_flag_requests[i]));
            } 
        }
    }
    return 0;
}

HYPRE_Int 
FindNeighborProcessors(hypre_ParCSRMatrix *A, 
   map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > &send_proc_dofs,
   map<HYPRE_Int, set<HYPRE_Int> > &starting_dofs, 
   set<HYPRE_Int> &recv_procs,
   HYPRE_Int level, HYPRE_Int max_distance, HYPRE_Int *communication_cost)
{
   
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // !!! Timing
   vector<chrono::duration<double>> timings(10);
   auto total_start = chrono::system_clock::now();
   auto part_start = chrono::system_clock::now();

   // Nodes to request from other processors. Note, requests are only issued to processors within distance 1, i.e. within the original communication stencil for A
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   map< HYPRE_Int, map<HYPRE_Int, vector<pair<HYPRE_Int, HYPRE_Int> > > > request_proc_dofs; // request_proc_dofs[proc to request from, i.e. recv_proc][destination_proc][dof global index][distance]
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++) request_proc_dofs[ hypre_ParCSRCommPkgRecvProc(commPkg,i) ];

   HYPRE_Int *add_flag = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRMatrixNumRows(A), HYPRE_MEMORY_HOST);
   HYPRE_Int *add_flag_requests = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)), HYPRE_MEMORY_HOST);

   // !!! Debug
   HYPRE_Int total_starting_dofs = 0;
   HYPRE_Int total_request_dofs = 0;
   HYPRE_Int total_proc_connections = 0;

   // Recursively search through the operator stencil to find longer distance neighboring dofs
   // Loop over destination processors
   for (auto dest_proc_it = starting_dofs.begin(); dest_proc_it != starting_dofs.end(); ++dest_proc_it)
   {
      HYPRE_Int destination_proc = dest_proc_it->first;
      // Loop over starting nodes for this proc
      
      // !!! Debug
      total_starting_dofs += dest_proc_it->second.size();

      for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
      {
         HYPRE_Int dof_index = *dof_it;
         add_flag[dof_index] = send_proc_dofs[destination_proc][dof_index]; // !!! Optimization: try to avoid this look up
      }
      for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
      {
         HYPRE_Int dof_index = *dof_it;
         HYPRE_Int distance = add_flag[dof_index];
         RecursivelyFindNeighborNodes(dof_index, distance-1, A, add_flag, add_flag_requests);
      }
      AddToSendAndRequestDofs(A, add_flag, add_flag_requests, send_proc_dofs[destination_proc], request_proc_dofs, destination_proc);
      memset(add_flag, 0, sizeof(HYPRE_Int)*hypre_ParCSRMatrixNumRows(A) );
      memset(add_flag_requests, 0, sizeof(HYPRE_Int)*hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)) );
   }
   // Clear the list of starting dofs
   starting_dofs.clear();

    // !!! Debug
    /* if (myid == 21) */
    /* { */
    /*     cout << "Total owned dofs = " << hypre_ParCSRMatrixNumRows(A) << ", num cols offd = " << hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)) << endl; */
    /*     for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it) // Iterate over recv procs */
    /*         for (auto req_proc_inner_it = req_proc_it->second.begin(); req_proc_inner_it != req_proc_it->second.end(); ++req_proc_inner_it) // Iterate over destinations */
    /*         { */
    /*            total_proc_connections++; */
    /*            total_request_dofs += req_proc_inner_it->second.size(); */
    /*         } */
    /*     cout << "total proc connections = " << total_proc_connections << ", num send procs = " << send_proc_dofs.size() << ", num starting procs = " << starting_dofs.size() << endl; */
    /*     cout << "Total starting dofs = " << total_starting_dofs << endl */
    /*        << "Total request dofs = " << total_request_dofs << endl; */
    /* } */


   // !!! Timing
   auto part_end = chrono::system_clock::now();
   timings[1] = part_end - part_start;
   part_start = chrono::system_clock::now();

   //////////////////////////////////////////////////
   // Communicate newly connected longer-distance processors to send procs: sending to current long distance send_procs and receiving from current long distance recv_procs
   //////////////////////////////////////////////////

   // Get the sizes
   hypre_MPI_Request *requests = hypre_CTAlloc(hypre_MPI_Request, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   hypre_MPI_Status *statuses = hypre_CTAlloc(hypre_MPI_Status, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   HYPRE_Int request_cnt = 0;

   HYPRE_Int *recv_sizes = hypre_CTAlloc(HYPRE_Int, recv_procs.size(), HYPRE_MEMORY_HOST);
   HYPRE_Int cnt = 0;
   for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
   {
      hypre_MPI_Irecv(&(recv_sizes[cnt++]), 1, HYPRE_MPI_INT, *recv_proc_it, 6, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   HYPRE_Int *send_sizes = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
   {
      for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
      {
         if (req_proc_it->second.find(send_proc_it->first) != req_proc_it->second.end()) send_sizes[cnt]++; 
      }
      hypre_MPI_Isend(&(send_sizes[cnt]), 1, HYPRE_MPI_INT, send_proc_it->first, 6, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(send_proc_dofs.size() + recv_procs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, send_proc_dofs.size() + recv_procs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;

   // Allocate and post the recvs
   HYPRE_Int **recv_buffers = hypre_CTAlloc(HYPRE_Int*, recv_procs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
   {
      recv_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, recv_sizes[cnt], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[cnt], recv_sizes[cnt], HYPRE_MPI_INT, *recv_proc_it, 7, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      cnt++;
   }
   // Setup and send the send buffers
   HYPRE_Int **send_buffers = hypre_CTAlloc(HYPRE_Int*, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
   {
      send_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, send_sizes[cnt], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 0;
      for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
      {
         if (req_proc_it->second.find(send_proc_it->first) != req_proc_it->second.end()) send_buffers[cnt][inner_cnt++] = req_proc_it->first; 
      }
      hypre_MPI_Isend(send_buffers[cnt], send_sizes[cnt], HYPRE_MPI_INT, send_proc_it->first, 7, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += send_sizes[cnt]*sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(send_proc_dofs.size() + recv_procs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Update recv_procs
   HYPRE_Int old_num_recv_procs = recv_procs.size();
   for (HYPRE_Int i = 0; i < old_num_recv_procs; i++)
   {
      for (HYPRE_Int j = 0; j < recv_sizes[i]; j++)
      {
         recv_procs.insert(recv_buffers[i][j]);
      }
   }

   // Clean up memory
   for (size_t i = 0; i < send_proc_dofs.size(); i++) hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   for (size_t i = 0; i < request_proc_dofs.size(); i++) hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   // !!! Timing
   part_end = chrono::system_clock::now();
   timings[2] = part_end - part_start;
   part_start = chrono::system_clock::now();

   //////////////////////////////////////////////////
   // Communicate request dofs to processors that I recv from: sending to request_procs and receiving from distance 1 send procs
   //////////////////////////////////////////////////

   // Count up the send size: 1 + sum_{destination_procs}(2 + 2*num_requested_dofs)
   // send_buffer = [num destination procs, [request info for proc], [request info for proc], ... ]
   // [request info for proc] = [proc id, num requested dofs, [(dof index, distance), (dof index, distance), ...] ]

   // Exchange message sizes
   send_sizes = hypre_CTAlloc(HYPRE_Int, request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   recv_sizes = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgNumSends(commPkg), HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      hypre_MPI_Irecv(&(recv_sizes[i]), 1, HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg,i), 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   cnt = 0;
   for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
   {
      send_sizes[cnt]++;
      for (auto dest_proc_it = req_proc_it->second.begin(); dest_proc_it != req_proc_it->second.end(); ++dest_proc_it)
      {
         send_sizes[cnt] += 2 + 2*dest_proc_it->second.size();
      }
      hypre_MPI_Isend(&(send_sizes[cnt]), 1, HYPRE_MPI_INT, req_proc_it->first, 4, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait on the recv sizes, then free and re-allocate the requests and statuses
   hypre_MPI_Waitall(hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);
   requests = hypre_CTAlloc(hypre_MPI_Request, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   statuses = hypre_CTAlloc(hypre_MPI_Status, hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   request_cnt = 0;

   // Allocate recv buffers and post the recvs
   recv_buffers = hypre_CTAlloc(HYPRE_Int*, hypre_ParCSRCommPkgNumSends(commPkg), HYPRE_MEMORY_HOST);
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      recv_buffers[i] = hypre_CTAlloc(HYPRE_Int, recv_sizes[i], HYPRE_MEMORY_HOST);
      hypre_MPI_Irecv(recv_buffers[i], recv_sizes[i], HYPRE_MPI_INT, hypre_ParCSRCommPkgSendProc(commPkg,i), 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
   }
   
   // Setup the send buffer and post the sends
   send_buffers = hypre_CTAlloc(HYPRE_Int*, request_proc_dofs.size(), HYPRE_MEMORY_HOST);
   cnt = 0;
   for (auto req_proc_it = request_proc_dofs.begin(); req_proc_it != request_proc_dofs.end(); ++req_proc_it)
   {
      send_buffers[cnt] = hypre_CTAlloc(HYPRE_Int, send_sizes[cnt], HYPRE_MEMORY_HOST);
      HYPRE_Int inner_cnt = 0;
      send_buffers[cnt][inner_cnt++] = req_proc_it->second.size();
      for (auto dest_proc_it = req_proc_it->second.begin(); dest_proc_it != req_proc_it->second.end(); ++dest_proc_it)
      {
         send_buffers[cnt][inner_cnt++] = dest_proc_it->first;
         send_buffers[cnt][inner_cnt++] = dest_proc_it->second.size();
         for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
         {
            send_buffers[cnt][inner_cnt++] = dof_it->first;
            send_buffers[cnt][inner_cnt++] = dof_it->second;
         }
      }
      hypre_MPI_Isend(send_buffers[cnt], send_sizes[cnt], HYPRE_MPI_INT, req_proc_it->first, 5, hypre_MPI_COMM_WORLD, &(requests[request_cnt++]));
      if (communication_cost)
      {
         communication_cost[level*10 + 0]++;
         communication_cost[level*10 + 1] += send_sizes[cnt]*sizeof(HYPRE_Int);
      }
      cnt++;
   }

   // Wait 
   hypre_MPI_Waitall(hypre_ParCSRCommPkgNumSends(commPkg) + request_proc_dofs.size(), requests, statuses);
   hypre_TFree(requests, HYPRE_MEMORY_HOST);
   hypre_TFree(statuses, HYPRE_MEMORY_HOST);

   // Update send_proc_dofs and starting_dofs 
   // Loop over send_proc's, i.e. the processors that we just received from 
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumSends(commPkg); i++)
   {
      cnt = 0;      
      HYPRE_Int num_destination_procs = recv_buffers[i][cnt++];
      for (HYPRE_Int destination_proc = 0; destination_proc < num_destination_procs; destination_proc++)
      {
         // Get destination proc id and the number of requested dofs
         HYPRE_Int proc_id = recv_buffers[i][cnt++];
         HYPRE_Int num_requested_dofs = recv_buffers[i][cnt++];

         // create new map for this destination proc if it doesn't already exist
         send_proc_dofs[proc_id];

         // Loop over the requested dofs for this destination proc
         for (HYPRE_Int j = 0; j < num_requested_dofs; j++)
         {
            // Get the local index for this dof on this processor
            HYPRE_Int req_dof_local_index = recv_buffers[i][cnt++] - hypre_ParCSRMatrixFirstRowIndex(A);

            // If we already have a this dof accounted for for this destination...
            if (send_proc_dofs[proc_id].find(req_dof_local_index) != send_proc_dofs[proc_id].end())
            {
               // ... but at a smaller distance, overwrite with new distance and add to starting_dofs
               if (send_proc_dofs[proc_id][req_dof_local_index] < recv_buffers[i][cnt])
               {
                  send_proc_dofs[proc_id][req_dof_local_index] = recv_buffers[i][cnt];
                  starting_dofs[proc_id].insert(req_dof_local_index);
               }
            } 
            // Otherwise, add this dof for this destination at this distance and add to starting_dofs
            else
            {
               send_proc_dofs[proc_id][req_dof_local_index] = recv_buffers[i][cnt];
               starting_dofs[proc_id].insert(req_dof_local_index);
            }
            cnt++;
         }
      }
   }

   // Clean up memory
   for (size_t i = 0; i < send_proc_dofs.size(); i++) hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   for (size_t i = 0; i < request_proc_dofs.size(); i++) hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buffers, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(send_sizes, HYPRE_MEMORY_HOST);

   // !!! Timing
   part_end = chrono::system_clock::now();
   timings[3] = part_end - part_start;
   auto total_end = chrono::system_clock::now();
   timings[0] = total_end - total_start;
   /* if (myid == 21) */
   /* { */
   /*     cout.precision(3); */
   /*     cout << "Total time " << timings[0].count() << endl */
   /*         << "Part 1 " << timings[1].count() << endl */
   /*         << "Part 2 " << timings[2].count() << endl */
   /*         << "Part 3 " << timings[3].count() << endl; */
   /* } */

   return 0;
}

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;
   HYPRE_Int               num_levels = hypre_ParCompGridCommPkgNumLevels(compGridCommPkg);

   HYPRE_Int max_distance = padding[level] + num_ghost_layers;

   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Get the default (distance 1) number of send and recv procs
   HYPRE_Int      num_sends = hypre_ParCSRCommPkgNumSends(commPkg);
   HYPRE_Int      num_recvs = hypre_ParCSRCommPkgNumRecvs(commPkg);

   // If num_sends and num_recvs are zero, then simply note that in compGridCommPkg and we are done
   if (num_sends == 0 && num_recvs == 0)
   {
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = 0;
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = 0;
      HYPRE_Int num_procs;
      hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
   }
   else
   {
      // Initialize send_proc_dofs and the starting_dofs (this is how we will track nodes to send to each proc until routine finishes)
      map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > send_proc_dofs; // send_proc_dofs[send_proc] = send_dofs, send_dofs[dof_index] = distance value
      map<HYPRE_Int, set<HYPRE_Int> > starting_dofs; // starting_dofs[send_proc] = vector of starting dofs for searching through stencil
      for (i = 0; i < num_sends; i++)
      {
         send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)]; // initialize the send procs as the keys in the outer map
         starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)];
         start = hypre_ParCSRCommPkgSendMapStart(commPkg,i);
         finish = hypre_ParCSRCommPkgSendMapStart(commPkg,i+1);
         for (j = start; j < finish; j++)
         {
            send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)][hypre_ParCSRCommPkgSendMapElmt(commPkg,j)] = max_distance;
            starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)].insert(hypre_ParCSRCommPkgSendMapElmt(commPkg,j));
         }
      }

      //Initialize the recv_procs
      set<HYPRE_Int> recv_procs;
      for (i = 0; i < num_recvs; i++) recv_procs.insert( hypre_ParCSRCommPkgRecvProc(commPkg,i) );

      // Iteratively communicate with longer and longer distance neighbors to grow the communication stencils
      for (i = 0; i < max_distance - 1; i++)
      {
         FindNeighborProcessors(A, send_proc_dofs, starting_dofs, recv_procs, level, max_distance, communication_cost);
      }
   
      // Use send_proc_dofs and recv_procs to generate relevant info for CompGridCommPkg
      // Set the number of send and recv procs
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = send_proc_dofs.size();
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = recv_procs.size();
      // Setup the list of send procs
      hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][cnt] = send_proc_it->first;
         cnt++;
      }
      // Setup the list of recv procs. NOTE: want to retain original commPkg ordering for recv procs with additional info after
      hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, recv_procs.size(), HYPRE_MEMORY_HOST);
      for (auto i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
      {
         hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][i] = hypre_ParCSRCommPkgRecvProc(commPkg,i);
      }
      cnt = hypre_ParCSRCommPkgNumRecvs(commPkg);
      for (auto recv_proc_it = recv_procs.begin(); recv_proc_it != recv_procs.end(); ++recv_proc_it)
      {
         bool skip = false;
         for (auto i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
         {
            if (*recv_proc_it == hypre_ParCSRCommPkgRecvProc(commPkg,i))
            {
               skip = true;
               break;
            }
         }
         if (!skip)
            hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[level][cnt++] = *recv_proc_it;
      }

      // Setup the send flag. NOTE: want to retain original commPkg ordering for send elmts with additional info after
      // !!! Optimization: must be a better way to enforce commPkg send ordering 
      hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int**, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int*, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      HYPRE_Int proc_cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         cnt = 0;
         hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt] = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST); 
         hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[level][proc_cnt] = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST); 

         hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level] = hypre_CTAlloc(HYPRE_Int, send_proc_it->second.size(), HYPRE_MEMORY_HOST); 
         hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg)[level][proc_cnt][level] = send_proc_it->second.size();
          
         // Check whether this proc was in the original commPkg
         bool original_commPkg = false;
         HYPRE_Int original_proc = 0;
         for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumSends(commPkg); original_proc++)
         {
            if (send_proc_it->first == hypre_ParCSRCommPkgSendProc(commPkg,original_proc))
            {
               original_commPkg = true;
               break;
            }
         }

         if (original_commPkg)
         {
            // First, add the original commPkg info
            for (auto dof_it = send_proc_it->second.begin(); dof_it != send_proc_it->second.end(); ++dof_it)
            {
               // Look for dof in original commPkg list
               for (auto i = hypre_ParCSRCommPkgSendMapStart(commPkg,original_proc); i < hypre_ParCSRCommPkgSendMapStart(commPkg,original_proc+1); i++)
               {
                  if (hypre_ParCSRCommPkgSendMapElmt(commPkg,i) == dof_it->first)
                  {
                     // !!! Optimization: can I just remove the dofs from the list as I copy them over? Makes the loop adding the remaining info much cheaper...
                     if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = -(dof_it->first + 1);
                     else hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = dof_it->first;
                     cnt++;
                     break;
                  }
               }
            }
            // Then, add the remaining info !!! Optimization: this nested loop is bad!
            for (auto dof_it = send_proc_it->second.begin(); dof_it != send_proc_it->second.end(); ++dof_it)
            {
               // Look for dof in original commPkg list
               bool skip = false;
               for (auto i = hypre_ParCSRCommPkgSendMapStart(commPkg,original_proc); i < hypre_ParCSRCommPkgSendMapStart(commPkg,original_proc+1); i++)
               {
                  if (hypre_ParCSRCommPkgSendMapElmt(commPkg,i) == dof_it->first)
                  {
                     skip = true;
                     break;
                  }
               }
               if (!skip)
               {
                  if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = -(dof_it->first + 1);
                  else hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = dof_it->first;
                  cnt++;
               }
            }
         }
         else
         {
            for (auto dof_it = send_proc_it->second.begin(); dof_it != send_proc_it->second.end(); ++dof_it)
            {
               if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = -(dof_it->first + 1);
               else hypre_ParCompGridCommPkgSendFlag(compGridCommPkg)[level][proc_cnt][level][cnt] = dof_it->first;
               cnt++;
            }
         }
         proc_cnt++;
      }
   }

   return 0;
}

HYPRE_Int
UnpackRecvBuffer( HYPRE_Int *recv_buffer, hypre_ParCompGrid **compGrid, 
      hypre_ParCSRCommPkg *commPkg,
      HYPRE_Int **A_tmp_info,
      hypre_ParCompGridCommPkg *compGridCommPkg,
      HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
      HYPRE_Int ****recv_map, HYPRE_Int ****recv_redundant_marker, HYPRE_Int ***num_recv_nodes, 
      HYPRE_Int *recv_map_send_buffer_size, HYPRE_Int current_level, HYPRE_Int num_levels,
      HYPRE_Int *nodes_added_on_level, HYPRE_Int buffer_number, HYPRE_Int *num_resizes, 
      HYPRE_Int symmetric )
{
   // recv_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind] ]

   HYPRE_Int            level, i, j, k;
   HYPRE_Int            num_psi_levels, row_size, level_start, add_node_cnt;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // initialize the counter
   HYPRE_Int            cnt = 0;

   // get the number of levels received
   num_psi_levels = recv_buffer[cnt++];

   // Init the recv_map_send_buffer_size !!! I think this can just be set a priori instead of counting it up in this function... !!!
   *recv_map_send_buffer_size = num_levels - current_level - 1;

   ////////////////////////////////////////////////////////////////////
   // Treat current_level specially: no redundancy here, and recv positions need to agree with original ParCSRCommPkg (extra comp grid points at the end)
   ////////////////////////////////////////////////////////////////////

   // Get the compgrid matrix, specifically the nonowned parts that will be added to
   hypre_ParCompGridMatrix *A = hypre_ParCompGridA(compGrid[current_level]);
   hypre_CSRMatrix *owned_offd = hypre_ParCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(A);

   // get the number of nodes on this level
   num_recv_nodes[current_level][buffer_number][current_level] = recv_buffer[cnt++];
   nodes_added_on_level[current_level] += num_recv_nodes[current_level][buffer_number][current_level];

   // if necessary, reallocate more space for nonowned dofs
   HYPRE_Int max_nonowned = hypre_CSRMatrixNumRows(nonowned_diag);
   HYPRE_Int start_extra_dofs = hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]);
   if (num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs > max_nonowned) 
   {
      num_resizes[3*current_level]++;
      HYPRE_Int new_size = ceil(1.5*max_nonowned);
      if (new_size < num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs) 
         new_size = num_recv_nodes[current_level][buffer_number][current_level] + start_extra_dofs;
      hypre_ParCompGridResize(compGrid[current_level], new_size, current_level != num_levels-1); // !!! Is there a better way to manage memory? !!!
   }

   // Get the original number of recv dofs in the ParCSRCommPkg (if this proc was recv'd from in original)   
   HYPRE_Int num_original_recv_dofs = 0;
   if (commPkg)
      if (buffer_number < hypre_ParCSRCommPkgNumRecvs(commPkg)) 
         num_original_recv_dofs = hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number+1) - hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number);

   // Skip over original commPkg recv dofs !!! Optimization: can avoid sending GIDs here
   HYPRE_Int remaining_dofs = num_recv_nodes[current_level][buffer_number][current_level] - num_original_recv_dofs;
   cnt += num_original_recv_dofs;

   // Setup the recv map on current level
   recv_map[current_level][buffer_number][current_level] = hypre_CTAlloc(HYPRE_Int, num_recv_nodes[current_level][buffer_number][current_level], HYPRE_MEMORY_HOST);
   for (i = 0; i < num_original_recv_dofs; i++)
   {
      recv_map[current_level][buffer_number][current_level][i] = i + hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number) + hypre_ParCompGridNumOwnedNodes(compGrid[current_level]);
   }

   // Unpack global indices and setup sort and invsort
   hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]) += remaining_dofs;
   HYPRE_Int *sort_map = hypre_ParCompGridNonOwnedSort(compGrid[current_level]);
   HYPRE_Int *inv_sort_map = hypre_ParCompGridNonOwnedInvSort(compGrid[current_level]);
   HYPRE_Int *new_inv_sort_map = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(nonowned_diag), HYPRE_MEMORY_HOST);
   HYPRE_Int sort_cnt = 0;
   HYPRE_Int compGrid_cnt = 0;
   HYPRE_Int incoming_cnt = 0;
   while (incoming_cnt < remaining_dofs && compGrid_cnt < start_extra_dofs)
   {
      // !!! Optimization: don't have to do these assignments every time... probably doesn't save much (i.e. only update incoming_global_index when necessary, etc.)
      HYPRE_Int incoming_global_index = recv_buffer[cnt];
      HYPRE_Int compGrid_global_index = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[current_level])[ inv_sort_map[compGrid_cnt] ];

      HYPRE_Int incoming_is_real = 1;
      if (incoming_global_index < 0) 
      {
         incoming_global_index = -(incoming_global_index + 1);
         incoming_is_real = 0;
      }

      if (incoming_global_index < compGrid_global_index)
      {
         // Set global index and real marker for incoming extra dof
         hypre_ParCompGridNonOwnedGlobalIndices(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] = incoming_global_index;
         hypre_ParCompGridNonOwnedRealMarker(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] = incoming_is_real;

         if (incoming_is_real)
            recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = incoming_cnt + start_extra_dofs + hypre_ParCompGridNumOwnedNodes(compGrid[current_level]);
         else
            recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = -(incoming_cnt + start_extra_dofs + hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + 1);

         sort_map[ incoming_cnt + start_extra_dofs ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = incoming_cnt + start_extra_dofs;
         sort_cnt++;
         incoming_cnt++;
         cnt++;
      }
      else
      {
         sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
         compGrid_cnt++;
         sort_cnt++;
      }
   }
   while (incoming_cnt < remaining_dofs)
   {
      HYPRE_Int incoming_global_index = recv_buffer[cnt];
      HYPRE_Int incoming_is_real = 1;
      if (incoming_global_index < 0) 
      {
         incoming_global_index = -(incoming_global_index + 1);
         incoming_is_real = 0;
      }

      hypre_ParCompGridNonOwnedGlobalIndices(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] = incoming_global_index;
      hypre_ParCompGridNonOwnedRealMarker(compGrid[current_level])[ incoming_cnt + start_extra_dofs ] = incoming_is_real;

      if (incoming_is_real)
         recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = incoming_cnt + start_extra_dofs + hypre_ParCompGridNumOwnedNodes(compGrid[current_level]);
      else
         recv_map[current_level][buffer_number][current_level][incoming_cnt + num_original_recv_dofs] = -(incoming_cnt + start_extra_dofs + hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + 1);

      sort_map[ incoming_cnt + start_extra_dofs ] = sort_cnt;
      new_inv_sort_map[sort_cnt] = incoming_cnt + start_extra_dofs;
      sort_cnt++;
      incoming_cnt++;
      cnt++;
   }
   while (compGrid_cnt < start_extra_dofs)
   {
      sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
      new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
      compGrid_cnt++;
      sort_cnt++;
   }

   hypre_TFree(inv_sort_map, hypre_ParCompGridMemoryLocation(compGrid[current_level]));
   hypre_ParCompGridNonOwnedInvSort(compGrid[current_level]) = new_inv_sort_map;

   // Unpack coarse global indices (need these for original commPkg recvs as well). 
   // NOTE: store global indices for now, will be adjusted to local indices during SetupLocalIndices
   if (current_level != num_levels-1)
   {
      for (i = 0; i < num_original_recv_dofs; i++)
      {
         HYPRE_Int coarse_index = recv_buffer[cnt++];
         if (coarse_index != -1) coarse_index = -(coarse_index+2); // Marking coarse indices that need setup by negative mapping
         hypre_ParCompGridNonOwnedCoarseIndices(compGrid[current_level])[i + hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number)] = coarse_index;
      }
      for (i = 0; i < remaining_dofs; i++)
      {
         HYPRE_Int coarse_index = recv_buffer[cnt++];
         if (coarse_index != -1) coarse_index = -(coarse_index+2); // Marking coarse indices that need setup by negative mapping
         hypre_ParCompGridNonOwnedCoarseIndices(compGrid[current_level])[i + start_extra_dofs] = coarse_index;
      }
   }

   // Unpack the col indices of A
   HYPRE_Int row_sizes_start = cnt;
   cnt += num_recv_nodes[current_level][buffer_number][current_level];

   // Setup col indices for original commPkg dofs
   for (i = 0; i < num_original_recv_dofs; i++)
   {
      HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number) + i ];
      HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number) + i ];

      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      for (j = 0; j < row_size; j++)
      {
         HYPRE_Int incoming_index = recv_buffer[cnt++];

         // Incoming is a global index (could be owned or nonowned)
         if (incoming_index < 0)
         {
            incoming_index = -(incoming_index+1);
            // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
            if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[current_level]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[current_level]))
            {
               // Add to offd
               if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                  hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd), hypre_CSRMatrixNumCols(nonowned_offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
               hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[current_level]);
            }
            else
            {
               // Add to diag (global index, not in buffer, so we store global index and get a local index during SetupLocalIndices)
               if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
               {
                  hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
                  hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), hypre_ParCompGridMemoryLocation(compGrid[current_level]));
               }
               hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level])[ hypre_ParCompGridNumMissingColIndices(compGrid[current_level])++ ] = diag_rowptr;
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index+1);
            }
         }
         // Incoming is an index to dofs within the buffer (by construction, nonowned)
         else
         {
            // Add to diag (index is within buffer, so we can directly go to local index)
            if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
            {
               hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
               hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), hypre_ParCompGridMemoryLocation(compGrid[current_level]));
            }
            if (incoming_index < num_original_recv_dofs)
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = incoming_index + hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number);
            else
            {
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = incoming_index - num_original_recv_dofs + start_extra_dofs;
            }
         }
      }

      // Update row pointers 
      hypre_CSRMatrixI(nonowned_diag)[ hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number) + i + 1 ] = diag_rowptr;
      hypre_CSRMatrixI(nonowned_offd)[ hypre_ParCSRCommPkgRecvVecStart(commPkg, buffer_number) + i + 1 ] = offd_rowptr;
   }

   // Temporary storage for extra comp grid dofs on this level (will be setup after all recv's during SetupLocalIndices)
   // A_tmp_info[buffer_number] = [ size, [row], size, [row], ... ]
   HYPRE_Int A_tmp_info_size = 2 + remaining_dofs;

   for (i = num_original_recv_dofs; i < num_recv_nodes[current_level][buffer_number][current_level]; i++)
   {
      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      A_tmp_info_size += row_size;
   }
   A_tmp_info[buffer_number] = hypre_CTAlloc(HYPRE_Int, A_tmp_info_size, hypre_ParCompGridMemoryLocation(compGrid[current_level]));
   HYPRE_Int A_tmp_info_cnt = 0;
   A_tmp_info[buffer_number][A_tmp_info_cnt++] = num_original_recv_dofs;
   A_tmp_info[buffer_number][A_tmp_info_cnt++] = remaining_dofs;
   for (i = num_original_recv_dofs; i < num_recv_nodes[current_level][buffer_number][current_level]; i++)
   {
      HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];
      A_tmp_info[buffer_number][A_tmp_info_cnt++] = row_size;
      for (j = 0; j < row_size; j++)
      {
         A_tmp_info[buffer_number][A_tmp_info_cnt++] = recv_buffer[cnt++];
      }
   }

   ////////////////////////////////////////////////////////////////////
   // loop over coarser psi levels
   ////////////////////////////////////////////////////////////////////

   for (level = current_level+1; level < current_level + num_psi_levels; level++)
   {
      // get the number of nodes on this level
      num_recv_nodes[current_level][buffer_number][level] = recv_buffer[cnt++];
      level_start = cnt;
      *recv_map_send_buffer_size += num_recv_nodes[current_level][buffer_number][level];

      A = hypre_ParCompGridA(compGrid[level]);
      owned_offd = hypre_ParCompGridMatrixOwnedOffd(A);
      nonowned_diag = hypre_ParCompGridMatrixNonOwnedDiag(A);
      nonowned_offd = hypre_ParCompGridMatrixNonOwnedOffd(A);

      HYPRE_Int num_nonowned = hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
      HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ num_nonowned ];
      HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ num_nonowned ];

      // Incoming nodes and existing (non-owned) nodes in the comp grid are both sorted by global index, so here we merge these lists together (getting rid of redundant nodes along the way)
      add_node_cnt = 0;

      // NOTE: Don't free incoming_dest because we set that as recv_map and use it outside this function
      HYPRE_Int *incoming_dest = hypre_CTAlloc(HYPRE_Int, num_recv_nodes[current_level][buffer_number][level], HYPRE_MEMORY_HOST);
      recv_redundant_marker[current_level][buffer_number][level] = hypre_CTAlloc(HYPRE_Int, num_recv_nodes[current_level][buffer_number][level], HYPRE_MEMORY_HOST);

      // if necessary, reallocate more space for compGrid
      if (num_recv_nodes[current_level][buffer_number][level] + num_nonowned > hypre_CSRMatrixNumRows(nonowned_diag)) 
      {
         num_resizes[3*level]++;
         HYPRE_Int new_size = ceil(1.5*hypre_CSRMatrixNumRows(nonowned_diag));
         if (new_size < num_recv_nodes[current_level][buffer_number][level] + num_nonowned) 
            new_size = num_recv_nodes[current_level][buffer_number][level] + num_nonowned;
         hypre_ParCompGridResize(compGrid[level], new_size, level != num_levels-1); // !!! Is there a better way to manage memory? !!!
      }

      sort_map = hypre_ParCompGridNonOwnedSort(compGrid[level]);
      inv_sort_map = hypre_ParCompGridNonOwnedInvSort(compGrid[level]);
      new_inv_sort_map = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(nonowned_diag), HYPRE_MEMORY_HOST);
      sort_cnt = 0;
      compGrid_cnt = 0;
      incoming_cnt = 0;
      HYPRE_Int dest = num_nonowned;

      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level] && compGrid_cnt < num_nonowned)
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int incoming_is_real = 1;
         if (incoming_global_index < 0) 
         {
            incoming_global_index = -(incoming_global_index + 1);
            incoming_is_real = 0;
         }

         // If incoming is owned, go on to the next
         if (incoming_global_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level]) && incoming_global_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level]))
         {
            recv_redundant_marker[current_level][buffer_number][level][incoming_cnt] = 1;
            if (incoming_is_real)
               incoming_dest[incoming_cnt] = incoming_global_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level]); // Save location info for use below
            else
               incoming_dest[incoming_cnt] = -(incoming_global_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // Save location info for use below
            incoming_cnt++;
            cnt++;
         }
         // Otherwise, merge
         else
         {
            HYPRE_Int compGrid_global_index = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ inv_sort_map[compGrid_cnt] ];

            if (incoming_global_index < compGrid_global_index)
            {
               sort_map[dest] = sort_cnt;
               new_inv_sort_map[sort_cnt] = dest;
               if (incoming_is_real)
                  incoming_dest[incoming_cnt] = dest + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               else
                  incoming_dest[incoming_cnt] = -(dest + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
               sort_cnt++;
               incoming_cnt++;
               dest++;
               cnt++;
               add_node_cnt++;
            }
            else if (incoming_global_index > compGrid_global_index)
            {
               sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
               new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
               compGrid_cnt++;
               sort_cnt++;
            }
            else
            {
               if (incoming_is_real && !hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[ inv_sort_map[compGrid_cnt] ])
               {
                  // !!! Symmetric: Need to insert A col ind (no space allocated for row info at ghost point... but now trying to overwrite with real dof)
                  hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[ inv_sort_map[compGrid_cnt] ] = 1;
                  incoming_dest[incoming_cnt] = inv_sort_map[compGrid_cnt] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); // Incoming real dof received to existing ghost location
                  incoming_cnt++;
                  cnt++;
               }
               else
               {
                  recv_redundant_marker[current_level][buffer_number][level][incoming_cnt] = 1;
                  if (incoming_is_real)
                     incoming_dest[incoming_cnt] = inv_sort_map[compGrid_cnt] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); // Save location info for use below
                  else
                     incoming_dest[incoming_cnt] = -(inv_sort_map[compGrid_cnt] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1); // Save location info for use below
                  incoming_cnt++;
                  cnt++;
               }
            }
         }
      }
      while (incoming_cnt < num_recv_nodes[current_level][buffer_number][level])
      {
         HYPRE_Int incoming_global_index = recv_buffer[cnt];
         HYPRE_Int incoming_is_real = 1;
         if (incoming_global_index < 0) 
         {
            incoming_global_index = -(incoming_global_index + 1);
            incoming_is_real = 0;
         }
         
         // If incoming is owned, go on to the next
         if (incoming_global_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level]) && incoming_global_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level]))
         {
            recv_redundant_marker[current_level][buffer_number][level][incoming_cnt] = 1;
            if (incoming_is_real) 
               incoming_dest[incoming_cnt] = incoming_global_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level]); // Save location info for use below
            else
               incoming_dest[incoming_cnt] = -(incoming_global_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // Save location info for use below
            incoming_cnt++;
            cnt++;
         }
         else
         {
            sort_map[dest] = sort_cnt;
            new_inv_sort_map[sort_cnt] = dest;
            if (incoming_is_real)
               incoming_dest[incoming_cnt] = dest + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            else
               incoming_dest[incoming_cnt] = -(dest + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
            sort_cnt++;
            incoming_cnt++;
            dest++;
            cnt++;
            add_node_cnt++;
         }
      }
      while (compGrid_cnt < num_nonowned)
      {
         sort_map[ inv_sort_map[compGrid_cnt] ] = sort_cnt;
         new_inv_sort_map[sort_cnt] = inv_sort_map[compGrid_cnt];
         compGrid_cnt++;
         sort_cnt++;
      }

      nodes_added_on_level[level] += add_node_cnt;

      // Free the old inv sort map and set new
      hypre_TFree(inv_sort_map, HYPRE_MEMORY_HOST);
      hypre_ParCompGridNonOwnedInvSort(compGrid[level]) = new_inv_sort_map;

      // Set recv_map[current_level] to incoming_dest
      recv_map[current_level][buffer_number][level] = incoming_dest;
      
      // Now copy in the new nodes to their appropriate positions
      cnt = level_start;
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++) 
      {   
         if (!recv_redundant_marker[current_level][buffer_number][level][i])
         {
            dest = incoming_dest[i];
            if (dest < 0) dest = -(dest+1);
            dest -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            HYPRE_Int global_index = recv_buffer[cnt];
            if (global_index < 0) 
            {
               global_index = -(global_index + 1);
               hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[ dest ] = 0;
            }
            else hypre_ParCompGridNonOwnedRealMarker(compGrid[level])[ dest ] = 1;
            hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ dest ] = global_index;
         }
         cnt++;
      }
      if (level != num_levels-1)
      {
         for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++) 
         {   
            if (!recv_redundant_marker[current_level][buffer_number][level][i])
            {
               dest = incoming_dest[i];
               if (dest < 0) dest = -(dest+1);
               dest -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               HYPRE_Int coarse_index = recv_buffer[cnt];
               if (coarse_index != -1) coarse_index = -(coarse_index+2); // Marking coarse indices that need setup by negative mapping
               hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[ dest ] = coarse_index;
            }
            cnt++;
         }
      }

      // Setup col indices 
      row_sizes_start = cnt;
      cnt += num_recv_nodes[current_level][buffer_number][level];
      for (i = 0; i < num_recv_nodes[current_level][buffer_number][level]; i++)
      {
         HYPRE_Int row_size = recv_buffer[ i + row_sizes_start ];

         // !!! Optimization: (probably small gain) right now, I disregard incoming info for real overwriting ghost (internal buf connectivity could be used to avoid a few binary searches later)
         // !!! Symmetric: need to insert col indices for ghosts overwritten as real somehow
         // if (incoming_dest[i] >= 0)
         dest = incoming_dest[i];
         if (dest < 0) dest = -(dest+1);
         dest -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);

         if (dest >= num_nonowned)
         {
            for (j = 0; j < row_size; j++)
            {
               HYPRE_Int incoming_index = recv_buffer[cnt++];

               // Incoming is a global index (could be owned or nonowned)
               if (incoming_index < 0)
               {
                  incoming_index = -(incoming_index+1);
                  // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
                  if (incoming_index >= hypre_ParCompGridFirstGlobalIndex(compGrid[level]) && incoming_index <= hypre_ParCompGridLastGlobalIndex(compGrid[level]))
                  {
                     // Add to offd
                     if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                        hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd), hypre_CSRMatrixNumCols(nonowned_offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
                     hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index - hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
                  }
                  else
                  {
                     // Add to diag (global index, not in buffer, so we store global index and get a local index during SetupLocalIndices)
                     if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                     {
                        hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
                        hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), hypre_ParCompGridMemoryLocation(compGrid[level]));
                     }
                     hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level])[ hypre_ParCompGridNumMissingColIndices(compGrid[level])++ ] = diag_rowptr;
                     hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index+1);
                  }
               }
               // Incoming is an index to dofs within the buffer (could be owned or nonowned)
               else
               {
                  HYPRE_Int local_index = incoming_dest[ incoming_index ];
                  if (local_index < 0) local_index = -(local_index + 1);

                  // Check whether dof is owned or nonowned
                  if (local_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
                  {
                     // Add to offd
                     if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                        hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd), hypre_CSRMatrixNumCols(nonowned_offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_offd) + 1));
                     hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = local_index;     
                  }
                  else
                  {
                     // Add to diag (index is within buffer, so we can directly go to local index)
                     if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                     {
                        hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1));
                        hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), hypre_ParCompGridMemoryLocation(compGrid[level]));
                     }
                     hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = local_index - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  }
               }
            }
            // Update row pointers 
            hypre_CSRMatrixI(nonowned_diag)[ dest + 1 ] = diag_rowptr;
            hypre_CSRMatrixI(nonowned_offd)[ dest + 1 ] = offd_rowptr;
         }
         else
         {
            cnt += row_size;
         }
      }

      hypre_ParCompGridNumNonOwnedNodes(compGrid[level]) += add_node_cnt;
   }

   return 0;
}

HYPRE_Int PackColIndCPU(HYPRE_Int *send_flag,
                        HYPRE_Int num_send_nodes,
                        HYPRE_Int *add_flag,
                        hypre_ParCompGrid *compGrid,
                        HYPRE_Int *send_buffer,
                        HYPRE_Int starting_cnt)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int i, j, send_elmt, add_flag_index;
   HYPRE_Int cnt = starting_cnt;
   HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid);
   for (i = 0; i < num_send_nodes; i++)
   {
      send_elmt = send_flag[i];
      if (send_elmt < 0) send_elmt = -(send_elmt + 1);

      // Owned point
      if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid))
      {
         hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid));
         hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid));
         // Get diag connections
         for (j = hypre_CSRMatrixI(diag)[send_elmt]; j < hypre_CSRMatrixI(diag)[send_elmt+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(diag)[j];
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid) + 1); // -(GID + 1)
            }
         }
         // Get offd connections
         for (j = hypre_CSRMatrixI(offd)[send_elmt]; j < hypre_CSRMatrixI(offd)[send_elmt+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridNumOwnedNodes(compGrid);
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[ hypre_CSRMatrixJ(offd)[j] ] + 1); // -(GID + 1)
            }
         }
      }
      // NonOwned point
      else if (send_elmt < total_num_nodes)
      {
         HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid);
         hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid));
         hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid));
         // Get diag connections
         for (j = hypre_CSRMatrixI(diag)[nonowned_index]; j < hypre_CSRMatrixI(diag)[nonowned_index+1]; j++)
         {
            if (hypre_CSRMatrixJ(diag)[j] >= 0)
            {
               add_flag_index = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridNumOwnedNodes(compGrid); 
               if (add_flag[add_flag_index] > 0)
               {
                  send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[ hypre_CSRMatrixJ(diag)[j] ] + 1); // -(GID + 1)
               }
            }
            else
            {
               send_buffer[cnt++] = hypre_CSRMatrixJ(diag)[j]; // -(GID + 1)
            }
         }
         // Get offd connections
         for (j = hypre_CSRMatrixI(offd)[nonowned_index]; j < hypre_CSRMatrixI(offd)[nonowned_index+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(offd)[j];
            if (add_flag[add_flag_index] > 0)
            {
               send_buffer[cnt++] = add_flag[add_flag_index] - 1; // Buffer connection
            }
            else
            {
               send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid) + 1); // -(GID + 1)
            }
         }
      }
      else send_flag[i] = send_elmt - total_num_nodes;
   }
   return cnt;
}

/* HYPRE_Int MarkCoarseListCPU(hypre_ParCompGrid *compGrid, */
/*                            hypre_ParCompGrid *compGrid_coarse, */
/*                            HYPRE_Int *send_flag, */
/*                            HYPRE_Int num_send_nodes, */
/*                            HYPRE_Int *starting_nodes) */
/* { */
/*    HYPRE_Int i; */
/*    HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid); */

/*    for (i = 0; i < num_send_nodes; i++) */
/*    { */
/*       HYPRE_Int send_elmt = send_flag[i]; */
/*       if (send_elmt >= total_num_nodes) */
/*          send_elmt -= total_num_nodes; */
/*       if (send_elmt >= 0) */
/*       { */
/*          if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid)) */
/*          { */
/*             HYPRE_Int coarse_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[send_elmt]; */
/*             if (coarse_index >= 0) starting_nodes[cnt++] = coarse_index; */
/*          } */
/*          else */
/*          { */
/*             HYPRE_Int coarse_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[send_elmt]; */
/*             if (coarse_index >= 0) starting_nodes[cnt++] = hypre_ParCompGridNonOwnedSort(compGrid_coarse)[coarse_index] + hypre_ParCompGridNumOwnedNodes(compGrid_coarse); */
/*          } */
/*       } */
/*    } */

/*    return cnt; */
/* } */

HYPRE_Int MarkCoarseCPU(HYPRE_Int *list,
           HYPRE_Int *marker,
           HYPRE_Int *owned_coarse_indices,
           HYPRE_Int *nonowned_coarse_indices,
           HYPRE_Int *sort_map,
           HYPRE_Int num_owned,
           HYPRE_Int total_num_nodes,
           HYPRE_Int num_owned_coarse,
           HYPRE_Int list_size,
           HYPRE_Int dist,
           HYPRE_Int use_sort,
           HYPRE_Int *nodes_to_add)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   HYPRE_Int i, coarse_index;
   for (i = 0; i < list_size; i++)
   {
      HYPRE_Int idx = list[i];
      if (idx >= 0)
      {
         if (idx >= total_num_nodes)
            idx -= total_num_nodes;
         if (idx < num_owned)
         {
            coarse_index = owned_coarse_indices[idx];
            if (coarse_index >= 0)
            {
                  // !!! Debug
                  /* if (myid == 10) printf("cpu idx %d, owned marker[%d] = %d\n", idx, coarse_index, dist); */
               marker[ coarse_index ] = dist;
               (*nodes_to_add) = 1;
            }
         }
         else
         {
            idx -= num_owned;
            coarse_index = nonowned_coarse_indices[idx];
            if (coarse_index >= 0)
            {
               if (use_sort) coarse_index = sort_map[ coarse_index ] + num_owned_coarse;
               else coarse_index = coarse_index + num_owned_coarse;
               marker[ coarse_index ] = dist;
               (*nodes_to_add) = 1;
            }
         }
      }
   }
   return 0;
}

HYPRE_Int* AddFlagToSendFlagCPU(hypre_ParCompGrid *compGrid,
                        HYPRE_Int *add_flag,
                        HYPRE_Int *num_send_nodes,
                        HYPRE_Int num_ghost_layers)
{
   HYPRE_Int i,cnt,add_flag_index;
   HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid);
   for (i = 0; i < total_num_nodes; i++)
   {
      if (add_flag[i] > 0)
      {
         (*num_send_nodes)++;
      }
   }

   HYPRE_Int *inv_sort_map = hypre_ParCompGridNonOwnedInvSort(compGrid);
   HYPRE_Int *send_flag = hypre_CTAlloc( HYPRE_Int, (*num_send_nodes), hypre_ParCompGridMemoryLocation(compGrid) );
   cnt =  0;
   i = 0;
   // First the nonowned indices coming before the owned block
   if (hypre_ParCompGridNumNonOwnedNodes(compGrid))
   {
      while (hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[inv_sort_map[i]] < hypre_ParCompGridFirstGlobalIndex(compGrid))
      {
         add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid);
         if (add_flag[add_flag_index] > num_ghost_layers)
         {
            send_flag[cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid);
            cnt++;
         }
         else if (add_flag[add_flag_index] > 0)
         {
            send_flag[cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid) + 1);
            cnt++;
         }
         i++;
         if (i == hypre_ParCompGridNumNonOwnedNodes(compGrid)) break;
      }
   }
   // Then the owned block
   for (add_flag_index = 0; add_flag_index < hypre_ParCompGridNumOwnedNodes(compGrid); add_flag_index++)
   {
      if (add_flag[add_flag_index] > num_ghost_layers)
      {
         send_flag[cnt] = add_flag_index;
         cnt++;
      }
      else if (add_flag[add_flag_index] > 0)
      {
         send_flag[cnt] = -(add_flag_index+1);
         cnt++;
      }
   }
   // Finally the nonowned indices coming after the owned block
   while (i < hypre_ParCompGridNumNonOwnedNodes(compGrid))
   {
      add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid);
      if (add_flag[add_flag_index] > num_ghost_layers)
      {
         send_flag[cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid);
         cnt++;
      }
      else if (add_flag[add_flag_index] > 0)
      {
         send_flag[cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid) + 1);
         cnt++;
      }
      i++;
   }

   return send_flag;
}

HYPRE_Int
SubtractLists(hypre_ParCompGrid *compGrid,
   HYPRE_Int *current_list, 
   HYPRE_Int *current_list_length, 
   HYPRE_Int *prev_list, 
   HYPRE_Int prev_list_length)
{
   // send_flag's are in global index ordering on each level, so can merge 
   HYPRE_Int prev_cnt = 0;
   HYPRE_Int current_cnt = 0;
   HYPRE_Int new_cnt = 0;
   while (current_cnt < (*current_list_length) && prev_cnt < prev_list_length)
   {
      // Get the global indices
      HYPRE_Int current_global_index = LocalToGlobalIndex(compGrid, current_list[current_cnt]);
      HYPRE_Int prev_global_index = LocalToGlobalIndex(compGrid, prev_list[prev_cnt]);

      // Do the merge
      if (current_global_index > prev_global_index)
      {
         prev_cnt++;
      }
      else if (current_global_index < prev_global_index)
      {
         current_list[new_cnt] = current_list[current_cnt];
         new_cnt++;
         current_cnt++;
      }
      else
      {
         // Special treatment for ghosts sent later as real
         if (prev_list[prev_cnt] < 0 && current_list[current_cnt] >= 0)
         {
            // This is the case of real dof sent to overwrite ghost. 
            // Current list is a positive local index here. Map beyond the range of total dofs to mark.
            if (current_list[current_cnt] < hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid))
               current_list[new_cnt] = current_list[current_cnt] + hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid);
            else
               current_list[new_cnt] = current_list[current_cnt];
            new_cnt++;
            current_cnt++;
            prev_cnt++;
         }
         else
         {
            prev_cnt++;
            current_cnt++;
         }
      }
   }
   while (current_cnt < (*current_list_length))
   {
      current_list[new_cnt] = current_list[current_cnt];
      new_cnt++;
      current_cnt++;
   }
   (*current_list_length) = new_cnt;

   return 0;
}

HYPRE_Int RemoveRedundancyCPU(hypre_ParAMGData* amg_data,
                              HYPRE_Int ****send_flag,
                              HYPRE_Int ***num_send_nodes,
                              hypre_ParCompGrid **compGrid,
                              hypre_ParCompGridCommPkg *compGridCommPkg,
                              HYPRE_Int current_level,
                              HYPRE_Int proc,
                              HYPRE_Int level)
{
   HYPRE_Int current_send_proc = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[current_level][proc];
   HYPRE_Int prev_proc, prev_level;
   HYPRE_Int num_send_nodes_before = num_send_nodes[current_level][proc][level];
   for (prev_level = current_level+1; prev_level <= level; prev_level++)
   {
      hypre_ParCSRCommPkg *original_commPkg = hypre_ParCSRMatrixCommPkg(hypre_ParAMGDataAArray(amg_data)[prev_level]);
      for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[prev_level]; prev_proc++)
      {
         if (hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
         {
            HYPRE_Int prev_list_end = num_send_nodes[prev_level][prev_proc][level];
            if (prev_level == level) 
            {
               HYPRE_Int original_proc;
               for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumSends(original_commPkg); original_proc++)
               {
                  if (hypre_ParCSRCommPkgSendProc(original_commPkg, original_proc) == current_send_proc) 
                  {
                     prev_list_end = hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgSendMapStart(original_commPkg, original_proc);
                     break;
                  }
               }
            }

            SubtractLists(compGrid[level],
               send_flag[current_level][proc][level], 
               &(num_send_nodes[current_level][proc][level]), 
               send_flag[prev_level][prev_proc][level], 
               prev_list_end);

            if (num_send_nodes[prev_level][prev_proc][level] - prev_list_end > 0)
            {
               SubtractLists(compGrid[level],
                  send_flag[current_level][proc][level], 
                  &(num_send_nodes[current_level][proc][level]), 
                  &(send_flag[prev_level][prev_proc][level][prev_list_end]), 
                  num_send_nodes[prev_level][prev_proc][level] - prev_list_end);
            }
         }
      }

      for (prev_proc = 0; prev_proc < hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[prev_level]; prev_proc++)
      {
         if (hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg)[prev_level][prev_proc] == current_send_proc)
         {
            HYPRE_Int prev_list_end = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level];
            if (prev_level == level) 
            {
               HYPRE_Int original_proc;
               for (original_proc = 0; original_proc < hypre_ParCSRCommPkgNumRecvs(original_commPkg); original_proc++)
               {
                  if (hypre_ParCSRCommPkgRecvProc(original_commPkg, original_proc) == current_send_proc) 
                  {
                     prev_list_end = hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc+1) - hypre_ParCSRCommPkgRecvVecStart(original_commPkg, original_proc);
                     break;
                  }
               }
            }

            SubtractLists(compGrid[level],
               send_flag[current_level][proc][level], 
               &(num_send_nodes[current_level][proc][level]), 
               hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level], 
               prev_list_end);

            if (hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end > 0)
            {
               SubtractLists(compGrid[level],
                  send_flag[current_level][proc][level], 
                  &(num_send_nodes[current_level][proc][level]), 
                  &(hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level][prev_list_end]), 
                  hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end);
            }
         }
      }
   }

   return 0;
}

HYPRE_Int*
PackSendBuffer(hypre_ParAMGData *amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int symmetric, HYPRE_Real *total_timings )
{
   // send_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind: either global indices or local col indices within buffer] ]

   // !!! Timing
   vector<chrono::duration<double>> timings(10);
   auto total_start = chrono::system_clock::now();

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int num_blocks;
   HYPRE_Int tpb = 64;
   HYPRE_Int level_switch_to_cpu = 10003;

   HYPRE_Int            level,i,j,k,cnt,row_length,send_elmt,coarse_grid_index,add_flag_index;
   HYPRE_Int            *nodes_to_add = hypre_CTAlloc(HYPRE_Int, 1, hypre_ParCompGridMemoryLocation(compGrid[current_level]));
   HYPRE_Int            **add_flag = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            num_psi_levels = 1;
   HYPRE_Int            coarse_proc;

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level - 1;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Mark the nodes to send (including Psi_c grid plus ghost nodes)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Count up the buffer size for the starting nodes
   add_flag[current_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]), hypre_ParCompGridMemoryLocation(compGrid[current_level]));

   (*buffer_size) += 2;
   if (current_level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level][proc][current_level];
   
   // !!! Timing
   auto inner_start = chrono::system_clock::now();

   for (i = 0; i < num_send_nodes[current_level][proc][current_level]; i++)
   {
      send_elmt = send_flag[current_level][proc][current_level][i];
      if (send_elmt < 0) send_elmt = -(send_elmt + 1);
      add_flag[current_level][send_elmt] = i + 1;

      hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[current_level]));
      hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[current_level]));
      (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
      (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
   }

   // !!! Timing
   auto inner_end = chrono::system_clock::now();
   timings[5] += inner_end - inner_start;
   inner_start = chrono::system_clock::now();

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != num_levels-1)
   {
      add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level+1]), hypre_ParCompGridMemoryLocation(compGrid[current_level+1]) );
#if defined(HYPRE_USING_GPU)
      if (current_level < level_switch_to_cpu-1)
      {
         num_blocks = num_send_nodes[current_level][proc][current_level]/tpb+1;
         MarkCoarseKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(send_flag[current_level][proc][current_level],
                 add_flag[current_level+1],
                 hypre_ParCompGridOwnedCoarseIndices(compGrid[current_level]),
                 hypre_ParCompGridNonOwnedCoarseIndices(compGrid[current_level]),
                 hypre_ParCompGridNumOwnedNodes(compGrid[current_level]),
                 hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]),
                 num_send_nodes[current_level][proc][current_level],
                 padding[current_level+1] + num_ghost_layers + 1,
                 nodes_to_add, myid);
         hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
         /* MarkCoarseGPU(compGrid[current_level], */
         /*                send_flag[current_level][proc][current_level], */
         /*                num_send_nodes[current_level][proc][current_level], */
         /*                add_flag[current_level+1], */ 
         /*                hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]), */
         /*                padding[current_level+1] + num_ghost_layers + 1); */
      }
      else
#endif
      {
         MarkCoarseCPU(send_flag[current_level][proc][current_level],
              add_flag[current_level+1],
              hypre_ParCompGridOwnedCoarseIndices(compGrid[current_level]),
              hypre_ParCompGridNonOwnedCoarseIndices(compGrid[current_level]),
              hypre_ParCompGridNonOwnedSort(compGrid[current_level+1]),
              hypre_ParCompGridNumOwnedNodes(compGrid[current_level]),
              hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]),
              hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]),
              num_send_nodes[current_level][proc][current_level],
              padding[current_level+1] + num_ghost_layers + 1,
              1,
              nodes_to_add);
      }
   }

   // !!! Timing
   inner_end = chrono::system_clock::now();
   timings[4] += inner_end - inner_start;


   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (*nodes_to_add)
      {
         num_psi_levels++;
         (*buffer_size)++;
         (*nodes_to_add) = 0;
         HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != num_levels-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]), hypre_ParCompGridMemoryLocation(compGrid[level+1]) );

         // !!! Timing
         inner_start = chrono::system_clock::now();
         
         // Expand by the padding on this level and add coarse grid counterparts if applicable
#if defined(HYPRE_USING_GPU)
         if (level < level_switch_to_cpu)
         {
            for (i = 0; i < padding[level] + num_ghost_layers; i++) 
            {
               num_blocks = hypre_ParCompGridNumOwnedNodes(compGrid[level])/tpb+1;
               ExpandPaddingNewKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(add_flag[level],
                           hypre_CSRMatrixJ( hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixJ( hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixI( hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixI( hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                           0,
                           hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                           padding[level] + num_ghost_layers + 1 - i); 
               num_blocks = hypre_ParCompGridNumNonOwnedNodes(compGrid[level])/tpb+1;
               ExpandPaddingNewKernel<<<num_blocks,tpb,0,HYPRE_STREAM(2)>>>(add_flag[level],
                           hypre_CSRMatrixJ( hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixJ( hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixI( hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_CSRMatrixI( hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level])) ),
                           hypre_ParCompGridNumNonOwnedNodes(compGrid[level]),
                           hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                           hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                           padding[level] + num_ghost_layers + 1 - i); 
               hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
               hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(2)));
            }
         }
         else
#endif
         {
            for (i = 0; i < total_num_nodes; i++)
            {
               if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
               else add_flag_index = hypre_ParCompGridNonOwnedSort(compGrid[level])[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

               if (add_flag[level][add_flag_index] == padding[level] + num_ghost_layers + 1)
                  RecursivelyBuildPsiComposite(i, padding[level] + num_ghost_layers, compGrid[level], add_flag[level], 1);
            }
         }

         // !!! Timing
         inner_end = chrono::system_clock::now();
         timings[1] += inner_end - inner_start;
         inner_start = chrono::system_clock::now();
         
#if defined(HYPRE_USING_GPU)
         if (level < level_switch_to_cpu)
         {
            send_flag[current_level][proc][level] = AddFlagToSendFlagGPU(compGrid[level],
                           add_flag[level],
                           &(num_send_nodes[current_level][proc][level]), num_ghost_layers);
         }
         else
#endif
         {
            send_flag[current_level][proc][level] = AddFlagToSendFlagCPU(compGrid[level],
                           add_flag[level],
                           &(num_send_nodes[current_level][proc][level]), num_ghost_layers);
         }

         // !!! Timing
         inner_end = chrono::system_clock::now();
         timings[2] += inner_end - inner_start;
         inner_start = chrono::system_clock::now();

         // Compare with previous send/recvs to eliminate redundant info
#if defined(HYPRE_USING_GPU)
         if (level < level_switch_to_cpu)
         {
            RemoveRedundancyGPU(amg_data,
                           compGrid,
                           compGridCommPkg,
                           send_flag,
                           hypre_ParCompGridCommPkgRecvMap(compGridCommPkg),
                           num_send_nodes,
                           current_level,
                           proc,
                           level,
                           HYPRE_STREAM(1)); 
         }
         else
#endif
         {
            RemoveRedundancyCPU(amg_data,
                           send_flag,
                           num_send_nodes,
                           compGrid,
                           compGridCommPkg,
                           current_level,
                           proc,
                           level);
         }

         // !!! Timing
         inner_end = chrono::system_clock::now();
         timings[3] += inner_end - inner_start;
         inner_start = chrono::system_clock::now();
         
         // Mark the points to start from on the next level
         if (level != num_levels-1)
         {
#if defined(HYPRE_USING_GPU)
            if (level < level_switch_to_cpu-1)
            {
               num_blocks = num_send_nodes[current_level][proc][level]/tpb+1;
               MarkCoarseKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(send_flag[current_level][proc][level],
                                      add_flag[level+1],
                                      hypre_ParCompGridOwnedCoarseIndices(compGrid[level]),
                                      hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level]),
                                      hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                                      hypre_ParCompGridNumOwnedNodes(compGrid[level+1]),
                                      num_send_nodes[current_level][proc][level],
                                      padding[level+1] + num_ghost_layers + 1,
                                      nodes_to_add, myid);
               hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
            }
            else
#endif
            {
               MarkCoarseCPU(send_flag[current_level][proc][level],
                 add_flag[level+1],
                 hypre_ParCompGridOwnedCoarseIndices(compGrid[level]),
                 hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level]),
                 hypre_ParCompGridNonOwnedSort(compGrid[level+1]),
                 hypre_ParCompGridNumOwnedNodes(compGrid[level]),
                 hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]),
                 hypre_ParCompGridNumOwnedNodes(compGrid[level+1]),
                 num_send_nodes[current_level][proc][level],
                 padding[level+1] + num_ghost_layers + 1,
                 1,
                 nodes_to_add);
            }
         }

         // !!! Timing
         inner_end = chrono::system_clock::now();
         timings[4] += inner_end - inner_start;
         inner_start = chrono::system_clock::now();

         // Count up the buffer sizes and adjust the add_flag 
         memset(add_flag[level], 0, sizeof(HYPRE_Int)*(hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level])) );
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][proc][level];

         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
            }
            else if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
            }
            else
            {
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
               add_flag[level][send_elmt] = i + 1;
            }
         }

         // !!! Timing
         inner_end = chrono::system_clock::now();
         timings[5] += inner_end - inner_start;
      }
      else break;
   }

   // !!! Timing
   auto total_end = chrono::system_clock::now();
   timings[0] = total_end - total_start;

   for (i = 0; i < 6; i++)
      total_timings[i] += timings[i].count();

   /* cout.precision(3); */
   /* // cout << scientific; */
   /* cout << "Rank " << myid << ", level " << current_level */
   /*    << ": total " << timings[0].count() */ 
   /*    << ", Expand " << timings[1].count() << " (" << 100 * (timings[1].count() / timings[0].count()) << "%)" */
   /*    << endl; */

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Pack the buffer
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_HOST);
   send_buffer[0] = num_psi_levels;
   cnt = 1;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      send_buffer[cnt++] = num_send_nodes[current_level][proc][level];

      // copy all global indices
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0)
         {
            send_elmt = -(send_elmt + 1);

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = -(send_elmt + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1);
            }
            else
            {
               send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ] + 1);
            }
         }
         else 
         {
            if (send_elmt >= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               send_buffer[cnt++] = send_elmt + hypre_ParCompGridFirstGlobalIndex(compGrid[level]);
            }
            else
            {
               send_buffer[cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]) ];
            }
         }
      }

      // if not on last level, copy coarse gobal indices
      if (level != num_levels-1)
      {
         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            else if (send_elmt >= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);

            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               if (hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ] >= 0)
                  send_buffer[cnt++] = hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ] + hypre_ParCompGridFirstGlobalIndex(compGrid[level+1]);
               else
                  send_buffer[cnt++] = hypre_ParCompGridOwnedCoarseIndices(compGrid[level])[ send_elmt ];
            }
            else 
            {
               HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               HYPRE_Int nonowned_coarse_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid[level])[ nonowned_index ];
               
               if (nonowned_coarse_index >= 0)
               {
                  send_buffer[cnt++] = hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level+1])[ nonowned_coarse_index ];
               }
               else if (nonowned_coarse_index == -1)
                  send_buffer[cnt++] = nonowned_coarse_index;
               else
                  send_buffer[cnt++] = -(nonowned_coarse_index+2);
            }
         }
      }

      // store the row length for matrix A
      for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      {
         send_elmt = send_flag[current_level][proc][level][i];
         if (send_elmt < 0) send_elmt = -(send_elmt + 1);
         if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
         {
            hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
            hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
            row_length = hypre_CSRMatrixI(diag)[ send_elmt + 1 ] - hypre_CSRMatrixI(diag)[ send_elmt ]
                       + hypre_CSRMatrixI(offd)[ send_elmt + 1 ] - hypre_CSRMatrixI(offd)[ send_elmt ];
         }
         else if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
            hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
            row_length = hypre_CSRMatrixI(diag)[ nonowned_index + 1 ] - hypre_CSRMatrixI(diag)[ nonowned_index ]
                       + hypre_CSRMatrixI(offd)[ nonowned_index + 1 ] - hypre_CSRMatrixI(offd)[ nonowned_index ];
         }
         else
         {
            row_length = 0;
            /* send_flag[current_level][proc][level][i] -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]); */
         }
         send_buffer[cnt++] = row_length;
      }

      // copy indices for matrix A (local connectivity within buffer where available, global index otherwise)
      cnt = PackColIndCPU(send_flag[current_level][proc][level],
                        num_send_nodes[current_level][proc][level],
                        add_flag[level],
                        compGrid[level],
                        send_buffer,
                        cnt);
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], hypre_ParCompGridMemoryLocation(compGrid[level]));
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid *compGrid, HYPRE_Int *add_flag, HYPRE_Int use_sort)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i,index,sort_index,coarse_grid_index;
   HYPRE_Int error_code = 0;

   HYPRE_Int *sort_map = hypre_ParCompGridNonOwnedSort(compGrid);
   /* HYPRE_Int node_gid, index_gid; */
   HYPRE_Int level = hypre_ParCompGridLevel(compGrid);

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   HYPRE_Int owned;
   if (node < hypre_ParCompGridNumOwnedNodes(compGrid))
   {
      owned = 1;
      diag = hypre_ParCompGridMatrixOwnedDiag( hypre_ParCompGridA(compGrid) );
      offd = hypre_ParCompGridMatrixOwnedOffd( hypre_ParCompGridA(compGrid) );
      /* node_gid = node + hypre_ParCompGridFirstGlobalIndex(compGrid); */
   }
   else
   {
      owned = 0;
      node = node - hypre_ParCompGridNumOwnedNodes(compGrid);
      diag = hypre_ParCompGridMatrixNonOwnedDiag( hypre_ParCompGridA(compGrid) );
      offd = hypre_ParCompGridMatrixNonOwnedOffd( hypre_ParCompGridA(compGrid) );      
      /* node_gid = hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[node]; */
   }

   // Look at neighbors in diag
   for (i = hypre_CSRMatrixI(diag)[node]; i < hypre_CSRMatrixI(diag)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(diag)[i];

      if (index >= 0)
      {
         if (owned)
         {
            sort_index = index;
            /* index_gid = index - hypre_ParCompGridFirstGlobalIndex(compGrid); */
         }
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
            /* index_gid = hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[index]; */
            index += hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            /* if (myid == 0 && level == 2) printf("Rank %d, level %d, node %d, index %d, sort_index %d, m %d\n", myid, level, node_gid, index_gid, sort_index, m); */
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, use_sort);
         }
      }
      else
      {
         error_code = 1;
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d, level %d, node gid %d: Error! Ran into a -1 index in diag when building Psi_c\n", 
            myid, hypre_ParCompGridLevel(compGrid), hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[node]);
      }
   }

   // Look at neighbors in offd
   for (i = hypre_CSRMatrixI(offd)[node]; i < hypre_CSRMatrixI(offd)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(offd)[i];

      if (index >= 0)
      {
         if (!owned) 
         {
            sort_index = index;
            /* index_gid = index + hypre_ParCompGridFirstGlobalIndex(compGrid); */
         }
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
            /* index_gid = hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[index]; */
            index += hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            /* if (myid == 0 && level == 2) printf("Rank %d, level %d, node %d, index %d, m %d\n", myid, level, node_gid, index_gid, m); */
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrid, add_flag, use_sort);
         }
      }
      else
      {
         error_code = 1; 
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d: Error! Ran into a -1 index in nonowned_offd when building Psi_c\n", myid);
      }
   }

   return error_code;
}

HYPRE_Int
LocalToGlobalIndex(hypre_ParCompGrid *compGrid, HYPRE_Int local_index)
{
   // Local index starts with 0 at beginning of owned dofs and continues through the nonowned (possible indices that are too large marking real overwriting ghost)
   if (local_index < 0) local_index = -(local_index+1);
   else if (local_index >= hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid))
      local_index -= hypre_ParCompGridNumOwnedNodes(compGrid) + hypre_ParCompGridNumNonOwnedNodes(compGrid);
   
   if (local_index < hypre_ParCompGridNumOwnedNodes(compGrid))
      return local_index + hypre_ParCompGridFirstGlobalIndex(compGrid);
   else
      return hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[local_index - hypre_ParCompGridNumOwnedNodes(compGrid)];
}



#endif
