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
   HYPRE_Int num_ghost_layers, HYPRE_Int symmetric );

HYPRE_Int RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid **compGrids, HYPRE_Int **add_flags,
                           HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int use_sort);


HYPRE_Int LocalToGlobalIndex(hypre_ParCompGrid *compGrid, HYPRE_Int local_index);

HYPRE_Int
RemoveRedundancy(hypre_ParCompGrid *compGrid,
   HYPRE_Int *current_list, 
   HYPRE_Int *current_list_length, 
   HYPRE_Int *prev_list, 
   HYPRE_Int prev_list_length);

#ifdef __cplusplus
}



#if defined(HYPRE_USING_GPU)
extern "C"
{
   __global__
   void PackColIndKernel(HYPRE_Int num_send_nodes, 
                        HYPRE_Int num_owned,
                        HYPRE_Int first_global_index,
                        HYPRE_Int *send_nodes,
                        HYPRE_Int *rowPtr,
                        HYPRE_Int *colInd,
                        HYPRE_Int *add_flag,
                        HYPRE_Int *offsets,
                        HYPRE_Int *global_ind,
                        HYPRE_Int *send_buffer)
   {
      HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
      HYPRE_Int j;
      if (i < num_send_nodes)
      {
         HYPRE_Int send_elmt = send_nodes[i];
         HYPRE_Int start = rowPtr[send_elmt];
         HYPRE_Int end = rowPtr[send_elmt+1];
         for (j = start; j < end; j++)
         {
            if (colInd[j] >= 0)
            {
               HYPRE_Int add_flag_index = colInd[j] + num_owned;
               if (add_flag[add_flag_index] > 0)
               {
                  send_buffer[offsets[i] + j - start] = add_flag[add_flag_index] - 1; // Buffer connection
               }
               else
               {
                  if (global_ind == NULL)
                  {
                     send_buffer[offsets[i] + j - start] = -(add_flag_index + first_global_index + 1); // -(GID + 1)
                  }
                  else
                  {
                     send_buffer[offsets[i] + j - start] = -(global_ind[ colInd[j] ] + 1); // -(GID + 1)
                  }
               }
            }
            else
            {
               send_buffer[offsets[i] + j - start] = colInd[j]; // -(GID + 1)
            }
         }
      } 
   }
}
#endif


HYPRE_Int
GetDofRecvProc(HYPRE_Int dof_index, HYPRE_Int neighbor_global_index, hypre_ParCSRMatrix *A)
{
   HYPRE_Int *colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int *offdRowPtr = hypre_CSRMatrixI( hypre_ParCSRMatrixOffd(A) );
   HYPRE_Int offdColIndex = -1;

   // Get the appropriate column index in the offd part of A
   for (HYPRE_Int i = offdRowPtr[dof_index]; i < offdRowPtr[dof_index+1]; i++)
   {
      if (colmap[ hypre_CSRMatrixJ( hypre_ParCSRMatrixOffd(A) )[i] ] == neighbor_global_index)
      {
         offdColIndex = hypre_CSRMatrixJ( hypre_ParCSRMatrixOffd(A) )[i];
      }
   }

   // Use that column index to find which processor this dof is received from
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int recv_proc = -1;
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++)
   {
      if (offdColIndex >= hypre_ParCSRCommPkgRecvVecStart(commPkg,i) && offdColIndex < hypre_ParCSRCommPkgRecvVecStart(commPkg,i+1)) 
         recv_proc = hypre_ParCSRCommPkgRecvProc(commPkg,i);
   }

   return recv_proc;
}

HYPRE_Int
RecursivelyFindNeighborNodes(HYPRE_Int dof_index, HYPRE_Int distance, hypre_ParCSRMatrix *A,
   map<HYPRE_Int, HYPRE_Int> &send_dofs, 
   map< HYPRE_Int, map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > > &request_proc_dofs, HYPRE_Int destination_proc )
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
      auto neighbor_dof = send_dofs.find(neighbor_index);
      if (neighbor_dof == send_dofs.end())
      {
         // If neighbor dof isn't in the send dofs, add it with appropriate distance and recurse
         send_dofs[neighbor_index] = distance;
         if (distance-1 > 0) RecursivelyFindNeighborNodes(neighbor_index, distance-1, A, send_dofs, request_proc_dofs, destination_proc);
      }
      else if (neighbor_dof->second < distance)
      {
         // If neighbor dof is in the send dofs, but at smaller distance, also need to update distance and recurse
         send_dofs[neighbor_index] = distance;
         if (distance-1 > 0) RecursivelyFindNeighborNodes(neighbor_index, distance-1, A, send_dofs, request_proc_dofs, destination_proc);
      }
   }
   // Look at offd neighbors
   for (i = hypre_CSRMatrixI(offd)[dof_index]; i < hypre_CSRMatrixI(offd)[dof_index+1]; i++)
   {
      HYPRE_Int neighbor_global_index = hypre_ParCSRMatrixColMapOffd(A)[ hypre_CSRMatrixJ(offd)[i] ];

      HYPRE_Int recv_proc = GetDofRecvProc(dof_index, neighbor_global_index, A);

      // If request proc isn't the destination proc
      if (recv_proc != destination_proc)
      {
         // Check whether we have already requested this node 
         auto req_dof = request_proc_dofs[recv_proc][destination_proc].find(neighbor_global_index);
         if (req_dof == request_proc_dofs[recv_proc][destination_proc].end())
         {
            // If this hasn't yet been requested, add it
            request_proc_dofs[recv_proc][destination_proc][neighbor_global_index] = distance;
         }
         else if (req_dof->second < distance)
         {
            // If reqest is already there, but at smaller distance, update the distance
            request_proc_dofs[recv_proc][destination_proc][neighbor_global_index] = distance;
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
   HYPRE_Int level, HYPRE_Int *communication_cost)
{
   
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   // Nodes to request from other processors. Note, requests are only issued to processors within distance 1, i.e. within the original communication stencil for A
   hypre_ParCSRCommPkg *commPkg = hypre_ParCSRMatrixCommPkg(A);
   map< HYPRE_Int, map<HYPRE_Int, map<HYPRE_Int, HYPRE_Int> > > request_proc_dofs; // request_proc_dofs[proc to request from, i.e. recv_proc][destination_proc][dof global index][distance]
   for (HYPRE_Int i = 0; i < hypre_ParCSRCommPkgNumRecvs(commPkg); i++) request_proc_dofs[ hypre_ParCSRCommPkgRecvProc(commPkg,i) ];

   // Recursively search through the operator stencil to find longer distance neighboring dofs
   // Loop over destination processors
   for (auto dest_proc_it = starting_dofs.begin(); dest_proc_it != starting_dofs.end(); ++dest_proc_it)
   {
      HYPRE_Int destination_proc = dest_proc_it->first;
      // Loop over starting nodes for this proc
      for (auto dof_it = dest_proc_it->second.begin(); dof_it != dest_proc_it->second.end(); ++dof_it)
      {
         HYPRE_Int dof_index = *dof_it;
         HYPRE_Int distance = send_proc_dofs[destination_proc][dof_index];
         RecursivelyFindNeighborNodes(dof_index, distance-1, A, send_proc_dofs[destination_proc], request_proc_dofs, destination_proc);
      }
   }
   // Clear the list of starting dofs
   starting_dofs.clear();

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

   return 0;
}

HYPRE_Int
SetupNearestProcessorNeighbors( hypre_ParCSRMatrix *A, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int level, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int *communication_cost )
{
   HYPRE_Int               i,j,cnt;
   HYPRE_Int               num_nodes = hypre_ParCSRMatrixNumRows(A);
   hypre_ParCSRCommPkg     *commPkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_Int               start,finish;

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
            send_proc_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)][hypre_ParCSRCommPkgSendMapElmt(commPkg,j)] = padding[level] + num_ghost_layers;
            starting_dofs[hypre_ParCSRCommPkgSendProc(commPkg,i)].insert(hypre_ParCSRCommPkgSendMapElmt(commPkg,j));
         }
      }

      //Initialize the recv_procs
      set<HYPRE_Int> recv_procs;
      for (i = 0; i < num_recvs; i++) recv_procs.insert( hypre_ParCSRCommPkgRecvProc(commPkg,i) );

      // Iteratively communicate with longer and longer distance neighbors to grow the communication stencils
      for (i = 0; i < padding[level] + num_ghost_layers - 1; i++)
      {
         FindNeighborProcessors(A, send_proc_dofs, starting_dofs, recv_procs, level, communication_cost);
      }
   
      // Use send_proc_dofs and recv_procs to generate relevant info for CompGridCommPkg
      // Set the number of send and recv procs
      hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level] = send_proc_dofs.size();
      hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level] = recv_procs.size();
      // Setup the list of send procs and count up the total number of send elmts.
      HYPRE_Int total_send_elmts = 0;
      hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size(), HYPRE_MEMORY_HOST);
      cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         hypre_ParCompGridCommPkgSendProcs(compGridCommPkg)[level][cnt] = send_proc_it->first;
         total_send_elmts += send_proc_it->second.size();
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

      // Setup the send map elmts, starts, and ghost marker. NOTE: want to retain original commPkg ordering for send elmts with additional info after
      // !!! Optimization: must be a better way to enforce commPkg send ordering 
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, send_proc_dofs.size() + 1, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
      hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level] = hypre_CTAlloc(HYPRE_Int, total_send_elmts, HYPRE_MEMORY_HOST);
      HYPRE_Int proc_cnt = 0;
      cnt = 0;
      for (auto send_proc_it = send_proc_dofs.begin(); send_proc_it != send_proc_dofs.end(); ++send_proc_it)
      {
         hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][proc_cnt++] = cnt;
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
                     hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][cnt] = dof_it->first;
                     if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][cnt] = 1;
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
                  hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][cnt] = dof_it->first;
                  if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][cnt] = 1;
                  cnt++;
               }
            }
         }
         else
         {
            for (auto dof_it = send_proc_it->second.begin(); dof_it != send_proc_it->second.end(); ++dof_it)
            {
               hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[level][cnt] = dof_it->first;
               if (dof_it->second <= num_ghost_layers) hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[level][cnt] = 1;
               cnt++;
            }
         }
      }
      hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[level][send_proc_dofs.size()] = total_send_elmts;
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

   hypre_TFree(inv_sort_map, HYPRE_MEMORY_HOST);
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
                  hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), HYPRE_MEMORY_HOST);
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
               hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), HYPRE_MEMORY_HOST);
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
   A_tmp_info[buffer_number] = hypre_CTAlloc(HYPRE_Int, A_tmp_info_size, HYPRE_MEMORY_HOST);
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
                        hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), HYPRE_MEMORY_HOST);
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
                        hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]) = hypre_TReAlloc(hypre_ParCompGridNonOwnedDiagMissingColIndices(compGrid[level]), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag) + 1), HYPRE_MEMORY_HOST);
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

HYPRE_Int*
PackSendBuffer(hypre_ParAMGData *amg_data, hypre_ParCompGrid **compGrid, hypre_ParCompGridCommPkg *compGridCommPkg, HYPRE_Int *buffer_size, 
   HYPRE_Int *send_flag_buffer_size, HYPRE_Int ****send_flag, HYPRE_Int ***num_send_nodes,
   HYPRE_Int proc, HYPRE_Int current_level, HYPRE_Int num_levels, HYPRE_Int *padding, HYPRE_Int num_ghost_layers, HYPRE_Int symmetric )
{
   // send_buffer = [ num_psi_levels , [level] , [level] , ... ]
   // level = [ num send nodes, [global indices] , [coarse global indices] , [A row sizes] , [A col ind: either global indices or local col indices within buffer] ]

   // !!! Timing
   // vector<chrono::duration<double>> timings(10);
   // auto total_start = chrono::system_clock::now();
   // auto time_start = chrono::system_clock::now();

   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int            level,i,j,k,cnt,row_length,send_elmt,coarse_grid_index,add_flag_index;
   HYPRE_Int            nodes_to_add = 0;
   HYPRE_Int            **add_flag = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            *num_owned_sends = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            *num_nonowned_sends = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **owned_sends = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **nonowned_sends = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **owned_diag_offsets = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **owned_offd_offsets = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **nonowned_diag_offsets = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            **nonowned_offd_offsets = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   HYPRE_Int            num_psi_levels = 1;
   HYPRE_Int            coarse_proc;

   // Get where to look in commPkgSendMapElmts
   HYPRE_Int            start = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc];
   HYPRE_Int            finish = hypre_ParCompGridCommPkgSendMapStarts(compGridCommPkg)[current_level][proc+1];

   // Get the sort maps
   HYPRE_Int            *sort_map;
   HYPRE_Int            *sort_map_coarse;

   // initialize send map buffer size
   (*send_flag_buffer_size) = num_levels - current_level - 1;

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Mark the nodes to send (including Psi_c grid plus ghost nodes)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Count up the buffer size for the starting nodes
   num_send_nodes[current_level][proc][current_level] = finish - start;
   send_flag[current_level][proc][current_level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_HOST );
   add_flag[current_level] = hypre_CTAlloc(HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level]), HYPRE_MEMORY_SHARED);

   num_owned_sends[current_level] = num_send_nodes[current_level][proc][current_level];
   num_nonowned_sends[current_level] = 0;
   owned_sends[current_level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_SHARED);
   owned_diag_offsets[current_level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_SHARED);
   owned_offd_offsets[current_level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][current_level], HYPRE_MEMORY_SHARED);

   HYPRE_Int *level_starts = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   level_starts[current_level] = 1;

   (*buffer_size) += 2;
   if (current_level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][current_level];
   else (*buffer_size) += 2*num_send_nodes[current_level][proc][current_level];

   for (i = start; i < finish; i++)
   {
      send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
      owned_sends[current_level][i - start] = send_elmt;
      if (hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         send_flag[current_level][proc][current_level][i - start] = -(send_elmt + 1);
      else
         send_flag[current_level][proc][current_level][i - start] = send_elmt;
      add_flag[current_level][send_elmt] = i - start + 1;

      hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[current_level]));
      hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[current_level]));
      owned_diag_offsets[current_level][i - start] = (*buffer_size);
      (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
      owned_offd_offsets[current_level][i - start] = (*buffer_size);
      (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
   }

   // Add the nodes listed by the coarse grid counterparts if applicable
   // Note that the compGridCommPkg is set up to list all nodes within the padding plus ghost layers
   if (current_level != num_levels-1)
   {
      add_flag[current_level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[current_level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[current_level+1]), HYPRE_MEMORY_SHARED );
      for (i = start; i < finish; i++)
      {
         // flag nodes that are repeated on the next coarse grid
         if (!hypre_ParCompGridCommPkgGhostMarker(compGridCommPkg)[current_level][i])
         {
            send_elmt = hypre_ParCompGridCommPkgSendMapElmts(compGridCommPkg)[current_level][i];
            coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid[current_level])[send_elmt];
            if ( coarse_grid_index != -1 ) 
            {
               add_flag[current_level+1][ coarse_grid_index ] = padding[current_level+1]+1;
               nodes_to_add = 1;
            }
         }
      }
   }

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Now build out the psi_c composite grid (along with required ghost nodes) on coarser levels
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   for (level = current_level + 1; level < num_levels; level++)
   {
      // if there are nodes to add on this grid
      if (nodes_to_add)
      {
         sort_map = hypre_ParCompGridNonOwnedSort(compGrid[level]);
         if (level != num_levels-1) sort_map_coarse = hypre_ParCompGridNonOwnedSort(compGrid[level+1]);
         HYPRE_Int *inv_sort_map = hypre_ParCompGridNonOwnedInvSort(compGrid[level]);
         
         num_psi_levels++;
         level_starts[level] = (*buffer_size);
         (*buffer_size)++;
         nodes_to_add = 0;

         // if we need coarse info, allocate space for the add flag on the next level
         if (level != num_levels-1) add_flag[level+1] = hypre_CTAlloc( HYPRE_Int, hypre_ParCompGridNumOwnedNodes(compGrid[level+1]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level+1]), HYPRE_MEMORY_SHARED );

         // Expand by the padding on this level and add coarse grid counterparts if applicable
         HYPRE_Int total_num_nodes = hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            if (add_flag[level][add_flag_index] == padding[level] + 1)
            {
               // Recursively add the region of padding (flagging coarse nodes on the next level if applicable)
               if (level != num_levels-1) RecursivelyBuildPsiComposite(i, padding[level], compGrid, add_flag, 1, &nodes_to_add, padding[level+1], level, 1);
               else RecursivelyBuildPsiComposite(i, padding[level], compGrid, add_flag, 0, NULL, 0, level, 1);
            }
         }

         // Expand by the number of ghost layers 
         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            if (add_flag[level][add_flag_index] > 1) add_flag[level][add_flag_index] = num_ghost_layers + 2;
            else if (add_flag[level][add_flag_index] == 1) add_flag[level][add_flag_index] = num_ghost_layers + 1;
         }

         for (i = 0; i < total_num_nodes; i++)
         {
            if (i < hypre_ParCompGridNumOwnedNodes(compGrid[level])) add_flag_index = i;
            else add_flag_index = sort_map[i - hypre_ParCompGridNumOwnedNodes(compGrid[level])] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);

            // Recursively add the region of ghost nodes (do not add any coarse nodes underneath)
            if (add_flag[level][add_flag_index] == num_ghost_layers + 1) RecursivelyBuildPsiComposite(i, num_ghost_layers, compGrid, add_flag, 0, NULL, 0, level, 1);
         }

         // Count up the total number of send nodes 
         for (i = 0; i < total_num_nodes; i++)
         {
            if (add_flag[level][i] > 0)
            {
               num_send_nodes[current_level][proc][level]++;
            }
         }

         // Save the indices (in global index ordering) 
         send_flag[current_level][proc][level] = hypre_CTAlloc( HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_HOST );
         cnt =  0;
         i = 0;
         // First the nonowned indices coming before the owned block
         if (hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            while (hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[inv_sort_map[i]] < hypre_ParCompGridFirstGlobalIndex(compGrid[level]))
            {
               add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               if (add_flag[level][add_flag_index] > num_ghost_layers)
               {
                  send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
                  cnt++;
               }
               else if (add_flag[level][add_flag_index] > 0)
               {
                  send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
                  cnt++;
               }
               i++;
               if (i == hypre_ParCompGridNumNonOwnedNodes(compGrid[level])) break;
            }
         }
         // Then the owned block
         for (add_flag_index = 0; add_flag_index < hypre_ParCompGridNumOwnedNodes(compGrid[level]); add_flag_index++)
         {
            if (add_flag[level][add_flag_index] > num_ghost_layers)
            {
               send_flag[current_level][proc][level][cnt] = add_flag_index;
               cnt++;
            }
            else if (add_flag[level][add_flag_index] > 0)
            {
               send_flag[current_level][proc][level][cnt] = -(add_flag_index+1);
               cnt++;
            }
         }
         // Finally the nonowned indices coming after the owned block
         while (i < hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
         {
            add_flag_index = i + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            if (add_flag[level][add_flag_index] > num_ghost_layers)
            {
               send_flag[current_level][proc][level][cnt] = inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               cnt++;
            }
            else if (add_flag[level][add_flag_index] > 0)
            {
               send_flag[current_level][proc][level][cnt] = -(inv_sort_map[i] + hypre_ParCompGridNumOwnedNodes(compGrid[level]) + 1);
               cnt++;
            }
            i++;
         }


         // !!! HERE


         // Eliminate redundant send info by comparing with previous send_flags and recv_maps
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

                  RemoveRedundancy(compGrid[level],
                     send_flag[current_level][proc][level], 
                     &(num_send_nodes[current_level][proc][level]), 
                     send_flag[prev_level][prev_proc][level], 
                     prev_list_end);

                  if (num_send_nodes[prev_level][prev_proc][level] - prev_list_end > 0)
                  {
                     RemoveRedundancy(compGrid[level],
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

                  RemoveRedundancy(compGrid[level],
                     send_flag[current_level][proc][level], 
                     &(num_send_nodes[current_level][proc][level]), 
                     hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level], 
                     prev_list_end);

                  if (hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end > 0)
                  {
                     RemoveRedundancy(compGrid[level],
                        send_flag[current_level][proc][level], 
                        &(num_send_nodes[current_level][proc][level]), 
                        &(hypre_ParCompGridCommPkgRecvMap(compGridCommPkg)[prev_level][prev_proc][level][prev_list_end]), 
                        hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg)[prev_level][prev_proc][level] - prev_list_end);
                  }
               }
            }
         }

         // Count up the buffer sizes and adjust the add_flag and get offsets
         memset(add_flag[level], 0, sizeof(HYPRE_Int)*(hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level])) );
         (*send_flag_buffer_size) += num_send_nodes[current_level][proc][level];
         if (level != num_levels-1) (*buffer_size) += 3*num_send_nodes[current_level][proc][level];
         else (*buffer_size) += 2*num_send_nodes[current_level][proc][level];

         owned_sends[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         nonowned_sends[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         owned_diag_offsets[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         owned_offd_offsets[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         nonowned_diag_offsets[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         nonowned_offd_offsets[level] = hypre_CTAlloc(HYPRE_Int, num_send_nodes[current_level][proc][level], HYPRE_MEMORY_SHARED);
         HYPRE_Int owned_cnt = 0;
         HYPRE_Int nonowned_cnt = 0;

         for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
         {
            send_elmt = send_flag[current_level][proc][level][i];
            if (send_elmt < 0) send_elmt = -(send_elmt + 1);
            if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               owned_sends[level][owned_cnt] = send_elmt;
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
               owned_diag_offsets[level][owned_cnt] = (*buffer_size);
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               owned_offd_offsets[level][owned_cnt] = (*buffer_size);
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
               owned_cnt++;
            }
            else if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]))
            {
               add_flag[level][send_elmt] = i + 1;
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               nonowned_sends[level][nonowned_cnt] = send_elmt;
               hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
               hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
               nonowned_diag_offsets[level][nonowned_cnt] = (*buffer_size);
               (*buffer_size) += hypre_CSRMatrixI(diag)[send_elmt+1] - hypre_CSRMatrixI(diag)[send_elmt];
               nonowned_offd_offsets[level][nonowned_cnt] = (*buffer_size);
               (*buffer_size) += hypre_CSRMatrixI(offd)[send_elmt+1] - hypre_CSRMatrixI(offd)[send_elmt];
               nonowned_cnt++;
            }
            else
            {
               send_elmt -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
               add_flag[level][send_elmt] = i + 1;
            }
         }
         num_owned_sends[level] = owned_cnt;
         num_nonowned_sends[level] = nonowned_cnt;
      }
      else break;
   }

   // !!! Timing
   // auto end = chrono::system_clock::now();
   // timings[1] = end - time_start;
   // time_start = chrono::system_clock::now();

   //////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Pack the buffer
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////

   HYPRE_Int *send_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_SHARED);
   send_buffer[0] = num_psi_levels;
   for (level = current_level; level < current_level + num_psi_levels; level++)
   {
      // store the number of nodes on this level
      cnt = level_starts[level];
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
         // !!! Symmetric optimization: if (send_elmt < 0 && symmetric) send_buffer[cnt++] = 0;
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
            send_flag[current_level][proc][level][i] -= hypre_ParCompGridNumOwnedNodes(compGrid[level]) + hypre_ParCompGridNumNonOwnedNodes(compGrid[level]);
         }
         send_buffer[cnt++] = row_length;
      }

      // !!! Timing
      auto inner_start = chrono::system_clock::now();

      // // copy indices for matrix A (local connectivity within buffer where available, global index otherwise)
      // for (i = 0; i < num_send_nodes[current_level][proc][level]; i++)
      // {
      //    send_elmt = send_flag[current_level][proc][level][i];
      //    // !!! Symmetric optimization: if (send_elmt < 0 && symmetric)
      //    if (send_elmt < 0) send_elmt = -(send_elmt + 1);

      //    // Owned point
      //    if (send_elmt < hypre_ParCompGridNumOwnedNodes(compGrid[level]))
      //    {
      //       hypre_CSRMatrix *diag = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
      //       hypre_CSRMatrix *offd = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
      //       // Get diag connections
      //       for (j = hypre_CSRMatrixI(diag)[send_elmt]; j < hypre_CSRMatrixI(diag)[send_elmt+1]; j++)
      //       {
      //          add_flag_index = hypre_CSRMatrixJ(diag)[j];
      //          if (add_flag[level][add_flag_index] > 0)
      //          {
      //             send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
      //          }
      //          else
      //          {
      //             send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
      //          }
      //       }
      //       // Get offd connections
      //       for (j = hypre_CSRMatrixI(offd)[send_elmt]; j < hypre_CSRMatrixI(offd)[send_elmt+1]; j++)
      //       {
      //          add_flag_index = hypre_CSRMatrixJ(offd)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
      //          if (add_flag[level][add_flag_index] > 0)
      //          {
      //             send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
      //          }
      //          else
      //          {
      //             send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(offd)[j] ] + 1); // -(GID + 1)
      //          }
      //       }
      //    }
      //    // NonOwned point
      //    else
      //    {
      //       HYPRE_Int nonowned_index = send_elmt - hypre_ParCompGridNumOwnedNodes(compGrid[level]);
      //       hypre_CSRMatrix *diag = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
      //       hypre_CSRMatrix *offd = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
      //       // Get diag connections
      //       for (j = hypre_CSRMatrixI(diag)[nonowned_index]; j < hypre_CSRMatrixI(diag)[nonowned_index+1]; j++)
      //       {
      //          if (hypre_CSRMatrixJ(diag)[j] >= 0)
      //          {
      //             add_flag_index = hypre_CSRMatrixJ(diag)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]); 
      //             if (add_flag[level][add_flag_index] > 0)
      //             {
      //                send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
      //             }
      //             else
      //             {
      //                send_buffer[cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(diag)[j] ] + 1); // -(GID + 1)
      //             }
      //          }
      //          else
      //          {
      //             send_buffer[cnt++] = hypre_CSRMatrixJ(diag)[j]; // -(GID + 1)
      //          }
      //       }
      //       // Get offd connections
      //       for (j = hypre_CSRMatrixI(offd)[nonowned_index]; j < hypre_CSRMatrixI(offd)[nonowned_index+1]; j++)
      //       {
      //          add_flag_index = hypre_CSRMatrixJ(offd)[j];
      //          if (add_flag[level][add_flag_index] > 0)
      //          {
      //             send_buffer[cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
      //          }
      //          else
      //          {
      //             send_buffer[cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
      //          }
      //       }
      //    }
      // }

// #if defined(HYPRE_USING_GPU)

//       const HYPRE_Int tpb=64;

//       hypre_CSRMatrix *mat = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
//       HYPRE_Int num_blocks=num_owned_sends[level]/tpb+1;
//       PackColIndKernel<<<num_blocks,tpb,0,HYPRE_STREAM(1)>>>(num_owned_sends[level], 
//                         0,
//                         hypre_ParCompGridFirstGlobalIndex(compGrid[level]),
//                         owned_sends[level],
//                         hypre_CSRMatrixI(mat),
//                         hypre_CSRMatrixJ(mat),
//                         add_flag[level],
//                         owned_diag_offsets[level],
//                         NULL,
//                         send_buffer);
//       mat = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
//       PackColIndKernel<<<num_blocks,tpb,0,HYPRE_STREAM(2)>>>(num_owned_sends[level], 
//                         hypre_ParCompGridNumOwnedNodes(compGrid[level]),
//                         hypre_ParCompGridFirstGlobalIndex(compGrid[level]),
//                         owned_sends[level],
//                         hypre_CSRMatrixI(mat),
//                         hypre_CSRMatrixJ(mat),
//                         add_flag[level],
//                         owned_offd_offsets[level],
//                         hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]),
//                         send_buffer);
//       mat = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
//       num_blocks=num_nonowned_sends[level]/tpb+1;
//       PackColIndKernel<<<num_blocks,tpb,0,HYPRE_STREAM(3)>>>(num_nonowned_sends[level], 
//                         hypre_ParCompGridNumOwnedNodes(compGrid[level]),
//                         hypre_ParCompGridFirstGlobalIndex(compGrid[level]),
//                         nonowned_sends[level],
//                         hypre_CSRMatrixI(mat),
//                         hypre_CSRMatrixJ(mat),
//                         add_flag[level],
//                         nonowned_diag_offsets[level],
//                         hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level]),
//                         send_buffer);
//       mat = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
//       PackColIndKernel<<<num_blocks,tpb,0,HYPRE_STREAM(4)>>>(num_nonowned_sends[level], 
//                         0,
//                         hypre_ParCompGridFirstGlobalIndex(compGrid[level]),
//                         nonowned_sends[level],
//                         hypre_CSRMatrixI(mat),
//                         hypre_CSRMatrixJ(mat),
//                         add_flag[level],
//                         nonowned_offd_offsets[level],
//                         NULL,
//                         send_buffer);

//       hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(1)));
//       hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(2)));
//       hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(3)));
//       hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
      
// #else

      // Pack owned diag col indices
      hypre_CSRMatrix *mat = hypre_ParCompGridMatrixOwnedDiag(hypre_ParCompGridA(compGrid[level]));
      for (i = 0; i < num_owned_sends[level]; i++)
      {
         send_elmt = owned_sends[level][i];
         HYPRE_Int col_ind_cnt = 0;
         for (j = hypre_CSRMatrixI(mat)[send_elmt]; j < hypre_CSRMatrixI(mat)[send_elmt+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(mat)[j];
            if (add_flag[level][add_flag_index] > 0)
               send_buffer[owned_diag_offsets[level][i] + col_ind_cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
            else
               send_buffer[owned_diag_offsets[level][i] + col_ind_cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
         }
      }
      // Pack owned offd col indices
      mat = hypre_ParCompGridMatrixOwnedOffd(hypre_ParCompGridA(compGrid[level]));
      for (i = 0; i < num_owned_sends[level]; i++)
      {
         send_elmt = owned_sends[level][i];
         HYPRE_Int col_ind_cnt = 0;
         for (j = hypre_CSRMatrixI(mat)[send_elmt]; j < hypre_CSRMatrixI(mat)[send_elmt+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(mat)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
            if (add_flag[level][add_flag_index] > 0)
               send_buffer[owned_offd_offsets[level][i] + col_ind_cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
            else
               send_buffer[owned_offd_offsets[level][i] + col_ind_cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(mat)[j] ] + 1); // -(GID + 1)
         }
      }
      // Pack nonowned diag col indices
      mat = hypre_ParCompGridMatrixNonOwnedDiag(hypre_ParCompGridA(compGrid[level]));
      for (i = 0; i < num_nonowned_sends[level]; i++)
      {
         send_elmt = nonowned_sends[level][i];
         HYPRE_Int col_ind_cnt = 0;
         for (j = hypre_CSRMatrixI(mat)[send_elmt]; j < hypre_CSRMatrixI(mat)[send_elmt+1]; j++)
         {
            if (hypre_CSRMatrixJ(mat)[j] >= 0)
            {
               add_flag_index = hypre_CSRMatrixJ(mat)[j] + hypre_ParCompGridNumOwnedNodes(compGrid[level]);
               if (add_flag[level][add_flag_index] > 0)
                  send_buffer[nonowned_diag_offsets[level][i] + col_ind_cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
               else
                  send_buffer[nonowned_diag_offsets[level][i] + col_ind_cnt++] = -(hypre_ParCompGridNonOwnedGlobalIndices(compGrid[level])[ hypre_CSRMatrixJ(mat)[j] ] + 1); // -(GID + 1)
            }
            else
            {
               send_buffer[nonowned_diag_offsets[level][i] + col_ind_cnt++] = hypre_CSRMatrixJ(mat)[j]; // -(GID + 1)
            }
         }
      }      
      // Pack nonowned offd col indices
      mat = hypre_ParCompGridMatrixNonOwnedOffd(hypre_ParCompGridA(compGrid[level]));
      for (i = 0; i < num_nonowned_sends[level]; i++)
      {
         send_elmt = nonowned_sends[level][i];
         HYPRE_Int col_ind_cnt = 0;
         for (j = hypre_CSRMatrixI(mat)[send_elmt]; j < hypre_CSRMatrixI(mat)[send_elmt+1]; j++)
         {
            add_flag_index = hypre_CSRMatrixJ(mat)[j];
            if (add_flag[level][add_flag_index] > 0)
               send_buffer[nonowned_offd_offsets[level][i] + col_ind_cnt++] = add_flag[level][add_flag_index] - 1; // Buffer connection
            else
               send_buffer[nonowned_offd_offsets[level][i] + col_ind_cnt++] = -(add_flag_index + hypre_ParCompGridFirstGlobalIndex(compGrid[level]) + 1); // -(GID + 1)
         }
      }

// #endif

      // !!! Timing
      // auto inner_end = chrono::system_clock::now();
      // timings[3] += inner_end - inner_start;
   }

   // Clean up memory
   for (level = 0; level < num_levels; level++)
   {
      if (add_flag[level]) hypre_TFree(add_flag[level], HYPRE_MEMORY_SHARED);
      if (owned_sends[level]) hypre_TFree(owned_sends[level], HYPRE_MEMORY_SHARED);
      if (nonowned_sends[level]) hypre_TFree(nonowned_sends[level], HYPRE_MEMORY_SHARED);
      if (owned_diag_offsets[level]) hypre_TFree(owned_diag_offsets[level], HYPRE_MEMORY_SHARED);
      if (owned_offd_offsets[level]) hypre_TFree(owned_offd_offsets[level], HYPRE_MEMORY_SHARED);
      if (nonowned_diag_offsets[level]) hypre_TFree(nonowned_diag_offsets[level], HYPRE_MEMORY_SHARED);
      if (nonowned_offd_offsets[level]) hypre_TFree(nonowned_offd_offsets[level], HYPRE_MEMORY_SHARED);
   }
   hypre_TFree(add_flag, HYPRE_MEMORY_HOST);
   
   hypre_TFree(num_owned_sends, HYPRE_MEMORY_HOST);
   hypre_TFree(num_nonowned_sends, HYPRE_MEMORY_HOST);
   hypre_TFree(owned_sends, HYPRE_MEMORY_HOST);
   hypre_TFree(nonowned_sends, HYPRE_MEMORY_HOST);
   hypre_TFree(owned_diag_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(owned_offd_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(nonowned_diag_offsets, HYPRE_MEMORY_HOST);
   hypre_TFree(nonowned_offd_offsets, HYPRE_MEMORY_HOST);


   // !!! Timing
   // end = chrono::system_clock::now();
   // timings[2] = end - time_start;
   // auto total_end = chrono::system_clock::now();
   // timings[0] = total_end - total_start;


   // !!! Timing: reference
   // auto ref_start = chrono::system_clock::now();
   // HYPRE_Int *test_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_HOST);
   // memcpy(test_buffer, send_buffer, (*buffer_size));
   // hypre_TFree(test_buffer, HYPRE_MEMORY_HOST);
   // auto ref_end = chrono::system_clock::now();
   // timings[4] = ref_end - ref_start;

   // ref_start = chrono::system_clock::now();
   // test_buffer = hypre_CTAlloc(HYPRE_Int, (*buffer_size), HYPRE_MEMORY_HOST);
   // for (i = 0; i < (*buffer_size); i++)
   // {
   //    test_buffer[i] = send_buffer[i];
   // }
   // hypre_TFree(test_buffer, HYPRE_MEMORY_HOST);
   // ref_end = chrono::system_clock::now();
   // timings[5] = ref_end - ref_start;


   // if (current_level == 0 && myid == 21)
   // if (myid == 21)
   // {
   //    cout.precision(3);
   //    // cout << scientific;
   //    cout << "Rank " << myid << ", level " << current_level
   //       // << ": total " << timings[0].count() 
   //       // << ", Build Psi_c " << timings[1].count() << " (" << 100 * (timings[1].count() / timings[0].count()) << "%)"
   //       << ", Reference " << timings[4].count() << " (" << 100 * (timings[4].count() / timings[4].count()) << "%)"
   //       << ", Reference 2 " << timings[5].count() << " (" << 100 * (timings[5].count() / timings[4].count()) << "%)"
   //       << ", Pack Buffer " << timings[2].count() << " (" << 100 * (timings[2].count() / timings[4].count()) << "%)"
   //       // << ", Pack Col Ind " << timings[3].count() << " (" << 100 * (timings[3].count() / timings[4].count()) << "%)"
   //       << endl;
   //    cout << "Rank " << myid << ", level " << current_level
   //       << ": total items packed " << cnt
   //       // << ", total col indices packed  " << total_col_indices_packed
   //       << endl;
   // }

   // Return the send buffer
   return send_buffer;
}

HYPRE_Int
RecursivelyBuildPsiComposite(HYPRE_Int node, HYPRE_Int m, hypre_ParCompGrid **compGrids, HYPRE_Int **add_flags,
                           HYPRE_Int need_coarse_info, HYPRE_Int *nodes_to_add, HYPRE_Int padding, HYPRE_Int level, HYPRE_Int use_sort)
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i,index,sort_index,coarse_grid_index;
   HYPRE_Int error_code = 0;

   hypre_ParCompGrid *compGrid = compGrids[level];
   HYPRE_Int *add_flag = add_flags[level];
   HYPRE_Int *sort_map = hypre_ParCompGridNonOwnedSort(compGrid);
   HYPRE_Int *add_flag_coarse = NULL;
   HYPRE_Int *sort_map_coarse = NULL;
   if (need_coarse_info)
   {
      add_flag_coarse = add_flags[level+1];
      sort_map_coarse = hypre_ParCompGridNonOwnedSort(compGrids[level+1]);
   }

   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   HYPRE_Int owned;
   if (node < hypre_ParCompGridNumOwnedNodes(compGrid))
   {
      owned = 1;
      diag = hypre_ParCompGridMatrixOwnedDiag( hypre_ParCompGridA(compGrid) );
      offd = hypre_ParCompGridMatrixOwnedOffd( hypre_ParCompGridA(compGrid) );
   }
   else
   {
      owned = 0;
      node = node - hypre_ParCompGridNumOwnedNodes(compGrid);
      diag = hypre_ParCompGridMatrixNonOwnedDiag( hypre_ParCompGridA(compGrid) );
      offd = hypre_ParCompGridMatrixNonOwnedOffd( hypre_ParCompGridA(compGrid) );      
   }

   // Look at neighbors in diag
   for (i = hypre_CSRMatrixI(diag)[node]; i < hypre_CSRMatrixI(diag)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(diag)[i];

      if (index >= 0)
      {
         if (owned) sort_index = index;
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrids, add_flags, need_coarse_info, nodes_to_add, padding, level, use_sort);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            if (owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[index];
            else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[index];

            if ( coarse_grid_index != -1 ) 
            {
               // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
               if (owned) sort_index = coarse_grid_index;
               else
               {
                  if (use_sort)
                  {
                     sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                  }
                  else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
               }
               add_flag_coarse[ sort_index ] = padding+1;
               *nodes_to_add = 1;
            }
         }
      }
      else
      {
         error_code = 1;
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d, level %d, node gid %d: Error! Ran into a -1 index in diag when building Psi_c\n", 
            myid, level, hypre_ParCompGridNonOwnedGlobalIndices(compGrid)[node]);
      }
   }

   // Look at neighbors in offd
   for (i = hypre_CSRMatrixI(offd)[node]; i < hypre_CSRMatrixI(offd)[node+1]; i++)
   {
      // Get the index of the neighbor
      index = hypre_CSRMatrixJ(offd)[i];

      if (index >= 0)
      {
         if (!owned) sort_index = index;
         else
         {
            if (use_sort) sort_index = sort_map[index] + hypre_ParCompGridNumOwnedNodes(compGrid);
            else sort_index = index + hypre_ParCompGridNumOwnedNodes(compGrid);
         }

         // If we still need to visit this index (note that add_flag[index] = m means we have already added all distance m-1 neighbors of index)
         if (add_flag[sort_index] < m)
         {
            add_flag[sort_index] = m;
            // Recursively call to find distance m-1 neighbors of index
            if (m-1 > 0) error_code = RecursivelyBuildPsiComposite(index, m-1, compGrids, add_flags, need_coarse_info, nodes_to_add, padding, level, use_sort);
         }
         // If m = 1, we won't do another recursive call, so make sure to flag the coarse grid here if applicable
         if (need_coarse_info && m == 1)
         {
            if (!owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[index];
            else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[index];

            if ( coarse_grid_index != -1 ) 
            {
               if (coarse_grid_index >= 0)
               {
                  // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
                  if (!owned) sort_index = coarse_grid_index;
                  else
                  {
                     if (use_sort) sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                     else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
                  }
                  add_flag_coarse[ sort_index ] = padding+1;
                  *nodes_to_add = 1;
               }
               else
               {
                  error_code = 1;
                  hypre_printf("Rank %d: Error! Ran into a coarse index that was not set up when building Psi_c\n", myid);
               }
            }
         }
      }
      else
      {
         error_code = 1; 
         if (owned == 1) hypre_printf("Rank %d: Error! Negative col index encountered in owned matrix\n");
         else hypre_printf("Rank %d: Error! Ran into a -1 index in nonowned_offd when building Psi_c\n", myid);
      }
   }

   // Flag this node on the next coarsest level if applicable
   if (need_coarse_info)
   {
      if (owned) coarse_grid_index = hypre_ParCompGridOwnedCoarseIndices(compGrid)[node];
      else coarse_grid_index = hypre_ParCompGridNonOwnedCoarseIndices(compGrid)[node];
      if ( coarse_grid_index != -1 ) 
      {
         // Again, need to set the add_flag to the appropriate value in order to recursively find neighbors on the next level
         if (owned) sort_index = coarse_grid_index;
         else
         {
            if (use_sort) sort_index = sort_map_coarse[coarse_grid_index] + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
            else sort_index = coarse_grid_index + hypre_ParCompGridNumOwnedNodes(compGrids[level+1]);
         }
         add_flag_coarse[ sort_index ] = padding+1;
         *nodes_to_add = 1;
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

HYPRE_Int
RemoveRedundancy(hypre_ParCompGrid *compGrid,
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


#endif
