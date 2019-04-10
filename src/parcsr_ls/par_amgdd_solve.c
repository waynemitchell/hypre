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

#define TEST_RES_COMM 0
#define DEBUGGING_MESSAGES 0

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_csr_block_matrix.h"   
#include "zfp.h"

HYPRE_Int
AddSolution( void *amg_vdata );

HYPRE_Real
GetCompositeResidual(hypre_ParCompGrid *compGrid);

HYPRE_Int
ZeroInitialGuess( void *amg_vdata );

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels );

HYPRE_Int
TestResComm(hypre_ParAMGData *amg_data);

HYPRE_Int
AgglomeratedProcessorsLocalResidualAllgather(hypre_ParAMGData *amg_data);

HYPRE_Int
MyZFPCompress(hypre_ParAMGData *amg_data, HYPRE_Complex *uncompressed_buffer, HYPRE_Int uncompressed_buffer_size, void **compressed_buffer, HYPRE_Int compressed_buffer_size, HYPRE_Int decompress, HYPRE_Real *zfp_errors);

HYPRE_Int
GetZFPFixedRateCompressedSizes(double rate, HYPRE_Complex *uncompressed_buffer, HYPRE_Int uncompressed_buffer_size);

HYPRE_Int 
hypre_BoomerAMGDDSolve( void *amg_vdata,
                                 hypre_ParCSRMatrix *A,
                                 hypre_ParVector *f,
                                 hypre_ParVector *u,
                                 HYPRE_Int *communication_cost,
                                 HYPRE_Real *zfp_errors )
{

   HYPRE_Int test_failed = 0;
   HYPRE_Int error_code;
   HYPRE_Int cycle_count = 0;
   HYPRE_Real alpha, beta;
   HYPRE_Real resid_nrm, resid_nrm_init, rhs_norm, relative_resid;

   // Get info from amg_data
   hypre_ParAMGData   *amg_data = amg_vdata;
   HYPRE_Real tol = hypre_ParAMGDataTol(amg_data);
   HYPRE_Int min_iter = hypre_ParAMGDataMinIter(amg_data);
   HYPRE_Int max_iter = hypre_ParAMGDataMaxIter(amg_data);
   HYPRE_Int converge_type = hypre_ParAMGDataConvergeType(amg_data);

   // Set the fine grid operator, left-hand side, and right-hand side
   hypre_ParAMGDataAArray(amg_data)[0] = A;
   hypre_ParAMGDataUArray(amg_data)[0] = u;
   hypre_ParAMGDataFArray(amg_data)[0] = f;

   // Store the original fine grid right-hand side in Vtemp and use f as the current fine grid residual
   hypre_ParVectorCopy(f, hypre_ParAMGDataVtemp(amg_data));
   alpha = -1.0;
   beta = 1.0;
   hypre_ParCSRMatrixMatvec(alpha, A, u, beta, f);

   // Setup convergence tolerance info
   if (tol > 0.)
   {
      resid_nrm = sqrt(hypre_ParVectorInnerProd(f,f));
      resid_nrm_init = resid_nrm;
      if (0 == converge_type)
      {
         rhs_norm = sqrt(hypre_ParVectorInnerProd(hypre_ParAMGDataVtemp(amg_data), hypre_ParAMGDataVtemp(amg_data)));
         if (rhs_norm)
         {
            relative_resid = resid_nrm_init / rhs_norm;
         }
         else
         {
            relative_resid = resid_nrm_init;
         }
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.;
   }

   // Main cycle loop
   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      // Do the AMGDD cycle
      error_code = hypre_BoomerAMGDD_Cycle(amg_vdata, communication_cost, zfp_errors);
      if (error_code) test_failed = 1;

      // Calculate a new resiudal
      if (tol > 0.)
      {
         hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), f);
         hypre_ParCSRMatrixMatvec(alpha, A, u, beta, f);
         resid_nrm = sqrt(hypre_ParVectorInnerProd(f,f));
         if (0 == converge_type)
         {
            if (rhs_norm)
            {
               relative_resid = resid_nrm / rhs_norm;
            }
            else
            {
               relative_resid = resid_nrm;
            }
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }
      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
   }

   // Copy RHS back into f
   hypre_ParVectorCopy(hypre_ParAMGDataVtemp(amg_data), f);

   return test_failed;
}

HYPRE_Int
hypre_BoomerAMGDD_Cycle( void *amg_vdata, HYPRE_Int *communication_cost, HYPRE_Real *zfp_errors )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int i,j,k,level;
   HYPRE_Int cycle_count = 0;
   hypre_ParAMGData  *amg_data = amg_vdata;
   hypre_ParCompGrid    **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int min_fac_iter = hypre_ParAMGDataMinFACIter(amg_data);
   HYPRE_Int max_fac_iter = hypre_ParAMGDataMaxFACIter(amg_data);
   HYPRE_Real fac_tol = hypre_ParAMGDataFACTol(amg_data);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Began AMG-DD cycle on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Form residual and do residual communication
   HYPRE_Int test_failed = 0;
   test_failed = hypre_BoomerAMGDDResidualCommunication( amg_vdata, communication_cost, zfp_errors );

   // Set zero initial guess for all comp grids on all levels
   ZeroInitialGuess( amg_vdata );

   // Setup convergence tolerance info
   HYPRE_Real resid_nrm = 1.;
   if (fac_tol != 0.) resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
   HYPRE_Real resid_nrm_init = resid_nrm;
   HYPRE_Real relative_resid = 1.;
   HYPRE_Real conv_fact = 0;
   
   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("About to do FAC cycles on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;
   for (level = 0; level < transition_level; level++)
   {
      if (!hypre_ParCompGridT( hypre_ParAMGDataCompGrid(amg_data)[level] )) 
         hypre_ParCompGridT( hypre_ParAMGDataCompGrid(amg_data)[level] ) = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridNumNodes(hypre_ParAMGDataCompGrid(amg_data)[level]), HYPRE_MEMORY_HOST);
      else for (i = 0; i < hypre_ParCompGridNumNodes(hypre_ParAMGDataCompGrid(amg_data)[level]); i++) hypre_ParCompGridT( hypre_ParAMGDataCompGrid(amg_data)[level] )[i] = 0.0;
      if (!hypre_ParCompGridS( hypre_ParAMGDataCompGrid(amg_data)[level] )) 
         hypre_ParCompGridS( hypre_ParAMGDataCompGrid(amg_data)[level] ) = hypre_CTAlloc(HYPRE_Complex, hypre_ParCompGridNumNodes(hypre_ParAMGDataCompGrid(amg_data)[level]), HYPRE_MEMORY_HOST);
      else for (i = 0; i < hypre_ParCompGridNumNodes(hypre_ParAMGDataCompGrid(amg_data)[level]); i++) hypre_ParCompGridS( hypre_ParAMGDataCompGrid(amg_data)[level] )[i] = 0.0;
   }

   // Do the cycles
   HYPRE_Int first_iteration = 1;
   if (fac_tol == 0.0)
   {
      while ( cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata, first_iteration );
         first_iteration = 0;

         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   else if (fac_tol > 0)
   {
      while ( (relative_resid >= fac_tol || cycle_count < min_fac_iter) && cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata, first_iteration );
         first_iteration = 0;

         // Check convergence and up the cycle count
         resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
         relative_resid = resid_nrm / resid_nrm_init;

         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   else if (fac_tol < 0)
   {
      fac_tol = -fac_tol;
      while ( (conv_fact <= fac_tol || conv_fact >= 1.0 || cycle_count < min_fac_iter) && cycle_count < max_fac_iter )
      {
         // Do FAC cycle
         hypre_BoomerAMGDD_FAC_Cycle( amg_vdata, first_iteration );
         first_iteration = 0;

         // Check convergence and up the cycle count
         resid_nrm = GetCompositeResidual(hypre_ParAMGDataCompGrid(amg_data)[0]);
         conv_fact = resid_nrm / resid_nrm_init;
         resid_nrm_init = resid_nrm;
         ++cycle_count;
         hypre_ParAMGDataNumFACIterations(amg_data) = cycle_count;
      }
   }
   


   // Update fine grid solution
   AddSolution( amg_vdata );

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished AMG-DD cycle on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   return test_failed;
}

HYPRE_Int
AddSolution( void *amg_vdata )
{
   hypre_ParAMGData  *amg_data = amg_vdata;
   HYPRE_Complex     *u = hypre_VectorData( hypre_ParVectorLocalVector( hypre_ParAMGDataUArray(amg_data)[0] ) );
   hypre_ParCompGrid    **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   HYPRE_Complex     *u_comp = hypre_ParCompGridU(compGrid[0]);
   HYPRE_Int         num_owned_nodes = hypre_ParCompGridOwnedBlockStarts(compGrid[0])[hypre_ParCompGridNumOwnedBlocks(compGrid[0])];
   HYPRE_Int         i;

   for (i = 0; i < num_owned_nodes; i++) u[i] += u_comp[i];

   return 0;
}

HYPRE_Real
GetCompositeResidual(hypre_ParCompGrid *compGrid)
{
   HYPRE_Int i,j;
   HYPRE_Real res_norm = 0.0;
   for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++)
   {
      if (hypre_ParCompGridARowPtr(compGrid)[i+1] - hypre_ParCompGridARowPtr(compGrid)[i] > 0)
      {
         HYPRE_Real res = hypre_ParCompGridF(compGrid)[i];
         for (j = hypre_ParCompGridARowPtr(compGrid)[i]; j < hypre_ParCompGridARowPtr(compGrid)[i+1]; j++)
         {
            res -= hypre_ParCompGridAData(compGrid)[j] * hypre_ParCompGridU(compGrid)[ hypre_ParCompGridAColInd(compGrid)[j] ];
         }
         res_norm += res*res;
      }
   }

   return sqrt(res_norm);
}

HYPRE_Int
ZeroInitialGuess( void *amg_vdata )
{
   HYPRE_Int   myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   hypre_ParAMGData  *amg_data = amg_vdata;
   HYPRE_Int i;

   HYPRE_Int level;
   for (level = 0; level < hypre_ParAMGDataNumLevels(amg_data); level++)
   {
      hypre_ParCompGrid    *compGrid = hypre_ParAMGDataCompGrid(amg_data)[level];
      for (i = 0; i < hypre_ParCompGridNumNodes(compGrid); i++) hypre_ParCompGridU(compGrid)[i] = 0.0;
   }
   
   return 0;
}

HYPRE_Int 
hypre_BoomerAMGDDResidualCommunication( void *amg_vdata, HYPRE_Int *communication_cost, HYPRE_Real *zfp_errors )
{
   HYPRE_Int   myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Began residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   MPI_Comm          comm;
   hypre_ParAMGData   *amg_data = amg_vdata;
   
   /* Data Structure variables */

   // level counters, indices, and parameters
   HYPRE_Int                  num_levels;
   HYPRE_Real                 alpha, beta;
   HYPRE_Int                  level,i,j;

   // info from amg
   hypre_ParCSRMatrix         **A_array;
   hypre_ParVector            **F_array;
   hypre_ParVector            **U_array;
   hypre_ParCSRMatrix         **P_array;
   hypre_ParVector            *Vtemp;
   HYPRE_Int                  *proc_first_index, *proc_last_index;
   HYPRE_Int                  *global_nodes;
   hypre_ParCompGrid          **compGrid;
   HYPRE_Int                  compress;

   // info from comp grid comm pkg
   hypre_ParCompGridCommPkg   *compGridCommPkg;
   HYPRE_Int                  num_send_procs, num_recv_procs, num_partitions;
   HYPRE_Int                  **send_procs;
   HYPRE_Int                  **recv_procs;
   HYPRE_Int                  **send_buffer_size;
   HYPRE_Int                  **recv_buffer_size;
   HYPRE_Int                  ***num_send_nodes;
   HYPRE_Int                  ***num_recv_nodes;
   HYPRE_Int                  ****send_flag;
   HYPRE_Int                  ****recv_map;

   // temporary arrays used for communication during comp grid setup
   HYPRE_Complex              **send_buffer;
   HYPRE_Complex              **recv_buffer;

   // temporary vectors used to copy data into composite grid structures
   hypre_Vector      *residual_local;
   HYPRE_Complex     *residual_data;

   // mpi stuff
   hypre_MPI_Request          *requests;
   hypre_MPI_Status           *status;
   HYPRE_Int                  request_counter = 0;

   // get info from amg
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);
   Vtemp = hypre_ParAMGDataVtemp(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   compGrid = hypre_ParAMGDataCompGrid(amg_data);
   compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);
   compress = hypre_ParAMGDataUseZFPCompression(amg_data);

   // get info from comp grid comm pkg
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;
   send_procs = hypre_ParCompGridCommPkgSendProcs(compGridCommPkg);
   recv_procs = hypre_ParCompGridCommPkgRecvProcs(compGridCommPkg);
   send_buffer_size = hypre_ParCompGridCommPkgSendBufferSize(compGridCommPkg);
   recv_buffer_size = hypre_ParCompGridCommPkgRecvBufferSize(compGridCommPkg);
   num_send_nodes = hypre_ParCompGridCommPkgNumSendNodes(compGridCommPkg);
   num_recv_nodes = hypre_ParCompGridCommPkgNumRecvNodes(compGridCommPkg);
   send_flag = hypre_ParCompGridCommPkgSendFlag(compGridCommPkg);
   recv_map = hypre_ParCompGridCommPkgRecvMap(compGridCommPkg);

   // get first and last global indices on each level for this proc
   proc_first_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   proc_last_index = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   global_nodes = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   for (level = 0; level < num_levels; level++)
   {
      proc_first_index[level] = hypre_ParVectorFirstIndex(F_array[level]);
      proc_last_index[level] = hypre_ParVectorLastIndex(F_array[level]);
      global_nodes[level] = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
   }

   // Restrict residual down to all levels (or just to the transition level) and initialize composite grids
   for (level = 0; level < transition_level-1; level++)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[level],F_array[level],
                            beta,F_array[level+1]);
   }
   if (transition_level != num_levels)
   {
      alpha = 1.0;
      beta = 0.0;
      hypre_ParCSRMatrixMatvecT(alpha,P_array[transition_level-1],F_array[transition_level-1],
                            beta,F_array[transition_level]);
   }

   // copy new restricted residual into comp grid structure
   HYPRE_Int local_myid = 0;
   for (level = 0; level < transition_level; level++)
   {
      // Check for agglomeration level
      if (hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level])
      {
         hypre_MPI_Comm_rank(hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level], &local_myid);
      }

      // Access the residual data
      residual_local = hypre_ParVectorLocalVector(F_array[level]);
      residual_data = hypre_VectorData(residual_local);
      for (i = hypre_ParCompGridOwnedBlockStarts(compGrid[level])[local_myid]; i < hypre_ParCompGridOwnedBlockStarts(compGrid[level])[local_myid+1]; i++)
      {
         hypre_ParCompGridF(compGrid[level])[i] = residual_data[i - hypre_ParCompGridOwnedBlockStarts(compGrid[level])[local_myid]];
      }
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("About to do coarse levels allgather on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   // Do Allgather of transition level 
   if (transition_level != num_levels)
   {
      residual_local = hypre_ParVectorLocalVector(F_array[transition_level]);
      residual_data = hypre_VectorData(residual_local);

      hypre_MPI_Allgatherv(residual_data, 
         hypre_VectorSize(residual_local), 
         HYPRE_MPI_COMPLEX, 
         hypre_ParCompGridF(compGrid[transition_level]), 
         hypre_ParCompGridCommPkgTransitionResRecvSizes(compGridCommPkg), 
         hypre_ParCompGridCommPkgTransitionResRecvDisps(compGridCommPkg), 
         HYPRE_MPI_COMPLEX, 
         hypre_MPI_COMM_WORLD);
   }

   // Do local allgathers for agglomerated procsesors
   AgglomeratedProcessorsLocalResidualAllgather(amg_data);


   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Entering loop over levels in residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   /* Outer loop over levels:
   Start from coarsest level and work up to finest */
   for (level = transition_level - 1; level >= 0; level--)
   {      
      // Get some communication info
      comm = hypre_ParCSRMatrixComm(A_array[level]);
      num_send_procs = hypre_ParCompGridCommPkgNumSendProcs(compGridCommPkg)[level];
      num_recv_procs = hypre_ParCompGridCommPkgNumRecvProcs(compGridCommPkg)[level];
      num_partitions = hypre_ParCompGridCommPkgNumPartitions(compGridCommPkg)[level];

      if ( num_send_procs || num_recv_procs ) // If there are any owned nodes on this level
      {
         // allocate space for the buffers, buffer sizes, requests and status, psiComposite_send, psiComposite_recv, send and recv maps
         recv_buffer = hypre_CTAlloc(HYPRE_Complex*, num_recv_procs, HYPRE_MEMORY_HOST);
         requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         send_buffer = hypre_CTAlloc(HYPRE_Complex*, num_partitions, HYPRE_MEMORY_HOST);
         request_counter = 0;

         // Setup extra arrays to hold compressed buffers, sizes, and MPI requests/statuses if using compression
         void **compressed_send_buffer;
         void **compressed_recv_buffer;
         HYPRE_Int *compressed_send_buffer_size;
         HYPRE_Int *compressed_recv_buffer_size;
         hypre_MPI_Request *size_requests;
         hypre_MPI_Status *size_status;
         HYPRE_Int size_request_counter = 0;
         if (compress)
         {
            compressed_send_buffer = hypre_CTAlloc(void*, num_partitions, HYPRE_MEMORY_HOST);
            compressed_recv_buffer = hypre_CTAlloc(void*, num_recv_procs, HYPRE_MEMORY_HOST);
            compressed_send_buffer_size = hypre_CTAlloc(HYPRE_Int, num_partitions, HYPRE_MEMORY_HOST);
            compressed_recv_buffer_size = hypre_CTAlloc(HYPRE_Int, num_recv_procs, HYPRE_MEMORY_HOST);
            size_requests = hypre_CTAlloc(hypre_MPI_Request, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
            size_status = hypre_CTAlloc(hypre_MPI_Status, num_send_procs + num_recv_procs, HYPRE_MEMORY_HOST );
         }

         // pack the send buffers
         for (i = 0; i < num_partitions; i++)
         {
            if (send_buffer_size[level][i])
            {
               send_buffer[i] = hypre_CTAlloc(HYPRE_Complex, send_buffer_size[level][i], HYPRE_MEMORY_HOST);
               PackResidualBuffer(send_buffer[i], send_flag[level][i], num_send_nodes[level][i], compGrid, level, num_levels);
               if (compress)
               {
                  // Compress the send buffer and get the compressed size
                  compressed_send_buffer_size[i] = MyZFPCompress(amg_data, send_buffer[i], send_buffer_size[level][i], &(compressed_send_buffer[i]), 0, 0, zfp_errors);
               }
            }
         }

         // if using fixed rate compression, get the compressed sizes
         if (compress == 1)
         {
            for (i = 0; i < num_recv_procs; i++)
               compressed_recv_buffer_size[i] = GetZFPFixedRateCompressedSizes(hypre_ParAMGDataZFPRate(amg_data), recv_buffer[i], recv_buffer_size[level][i]);
         }
         // otherwise when using compression, need to communicate the compressed sizes
         else if (compress > 1)
         {
            // post the sends for the sizes
            for (i = 0; i < num_send_procs; i++)
            {
               HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
               if (send_buffer_size[level][buffer_index])
               {
                  hypre_MPI_Isend(&(compressed_send_buffer_size[buffer_index]), 1, HYPRE_MPI_INT, send_procs[level][i], 4, comm, &size_requests[size_request_counter++]);
                  if (communication_cost)
                  {
                     communication_cost[level*7 + 4]++;
                     communication_cost[level*7 + 5] += sizeof(HYPRE_Int);
                  }
               }
            }

            // post the recvs for the sizes
            for (i = 0; i < num_recv_procs; i++)
            {
               if (recv_buffer_size[level][i])
               {
                  hypre_MPI_Irecv( &(compressed_recv_buffer_size[i]), 1, HYPRE_MPI_INT, recv_procs[level][i], 4, comm, &size_requests[size_request_counter++]);
               }
            }

            // wait on the sizes to be received
            hypre_MPI_Waitall( size_request_counter, size_requests, size_status );
            hypre_TFree(size_requests, HYPRE_MEMORY_HOST);
            hypre_TFree(size_status, HYPRE_MEMORY_HOST);
         }

         // allocate space for the receive buffers and post the receives
         for (i = 0; i < num_recv_procs; i++)
         {
            if (recv_buffer_size[level][i])
            {
               recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, recv_buffer_size[level][i], HYPRE_MEMORY_HOST );
               if (compress)
               {
                  compressed_recv_buffer[i] = hypre_CTAlloc(HYPRE_Complex, compressed_recv_buffer_size[i], HYPRE_MEMORY_HOST);
                  hypre_MPI_Irecv( compressed_recv_buffer[i], compressed_recv_buffer_size[i], MPI_BYTE, recv_procs[level][i], 3, comm, &requests[request_counter++]);
               }
               else hypre_MPI_Irecv( recv_buffer[i], recv_buffer_size[level][i], HYPRE_MPI_COMPLEX, recv_procs[level][i], 3, comm, &requests[request_counter++]);
            }
         }

         // post the sends
         for (i = 0; i < num_send_procs; i++)
         {
            HYPRE_Int buffer_index = hypre_ParCompGridCommPkgSendProcPartitions(compGridCommPkg)[level][i];
            if (send_buffer_size[level][buffer_index])
            {
               if (compress)
               {
                  hypre_MPI_Isend(compressed_send_buffer[buffer_index], compressed_send_buffer_size[buffer_index], MPI_BYTE, send_procs[level][i], 3, comm, &requests[request_counter++]);
                  if (communication_cost) communication_cost[level*7 + 5] += compressed_send_buffer_size[buffer_index];
               }
               else hypre_MPI_Isend(send_buffer[buffer_index], send_buffer_size[level][buffer_index], HYPRE_MPI_COMPLEX, send_procs[level][i], 3, comm, &requests[request_counter++]);
            }
         }

         // wait for buffers to be received
         hypre_MPI_Waitall( request_counter, requests, status );

         hypre_TFree(requests, HYPRE_MEMORY_HOST);
         hypre_TFree(status, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_partitions; i++)
         {
            hypre_TFree(send_buffer[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(send_buffer, HYPRE_MEMORY_HOST);
         if (compress)
         {
            for (i = 0; i < num_partitions; i++)
            {
               hypre_TFree(compressed_send_buffer[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(compressed_send_buffer, HYPRE_MEMORY_HOST);
            hypre_TFree(compressed_send_buffer_size, HYPRE_MEMORY_HOST);
         }

         // loop over received buffers
         for (i = 0; i < num_recv_procs; i++)
         {
            if (recv_buffer_size[level][i])
            {
               // if necessary, decompress the recv buffer
               if (compress)
               {
                  MyZFPCompress(amg_data, recv_buffer[i], recv_buffer_size[level][i], &(compressed_recv_buffer[i]), compressed_recv_buffer_size[i], 1, NULL);
               }
               // unpack the buffers
               UnpackResidualBuffer(recv_buffer[i], recv_map[level][i], num_recv_nodes[level][i], compGrid, level, num_levels);
            }
         }

         // clean up memory for this level
         for (i = 0; i < num_recv_procs; i++)
         {
            hypre_TFree(recv_buffer[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(recv_buffer, HYPRE_MEMORY_HOST);
         if (compress)
         {
            for (i = 0; i < num_recv_procs; i++)
            {
               hypre_TFree(compressed_recv_buffer[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(compressed_recv_buffer, HYPRE_MEMORY_HOST);
            hypre_TFree(compressed_recv_buffer_size, HYPRE_MEMORY_HOST);
         }
      }

      #if DEBUGGING_MESSAGES
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      if (myid == 0) hypre_printf("   Finished residual communication on level %d on all ranks\n", level);
      hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
      #endif
   }

   #if DEBUGGING_MESSAGES
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   if (myid == 0) hypre_printf("Finished residual communication on all ranks\n");
   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);
   #endif

   #if TEST_RES_COMM
   HYPRE_Int test_failed = TestResComm(amg_data);
   #endif

   // Cleanup memory
   hypre_TFree(proc_first_index, HYPRE_MEMORY_HOST);
   hypre_TFree(proc_last_index, HYPRE_MEMORY_HOST);
   hypre_TFree(global_nodes, HYPRE_MEMORY_HOST);
   
   #if TEST_RES_COMM
   return test_failed;
   #else
   return 0;
   #endif
}

HYPRE_Int
PackResidualBuffer( HYPRE_Complex *send_buffer, HYPRE_Int **send_flag, HYPRE_Int *num_send_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   HYPRE_Int                  level,i,cnt = 0;

   // pack the send buffer
   for (level = current_level; level < num_levels; level++)
   {
      for (i = 0; i < num_send_nodes[level]; i++)
      {
         send_buffer[cnt++] = hypre_ParCompGridF(compGrid[level])[ send_flag[level][i] ];
      }
   }

   return 0;

}

HYPRE_Int
UnpackResidualBuffer( HYPRE_Complex *recv_buffer, HYPRE_Int **recv_map, HYPRE_Int *num_recv_nodes, hypre_ParCompGrid **compGrid, HYPRE_Int current_level, HYPRE_Int num_levels)
{
   HYPRE_Int                  level,i,cnt = 0, map_cnt, num_nodes;

   // loop over levels
   for (level = current_level; level < num_levels; level++)
   {
      for (i = 0; i < num_recv_nodes[level]; i++) 
      {
         hypre_ParCompGridF(compGrid[level])[ recv_map[level][i] ] = recv_buffer[cnt++];
      }
   }

   return 0;
}

HYPRE_Int
TestResComm(hypre_ParAMGData *amg_data)
{
   // Get MPI info
   HYPRE_Int myid, num_procs;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);

   // Get info from the amg data structure
   hypre_ParCompGrid **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(hypre_ParAMGDataCompGridCommPkg(amg_data));
   if (transition_level < 0) transition_level = num_levels;

   HYPRE_Int test_failed = 0;

   // For each processor and each level broadcast the residual data and global indices out and check agains the owning procs
   HYPRE_Int proc;
   HYPRE_Int i;
   for (proc = 0; proc < num_procs; proc++)
   {
      HYPRE_Int level;
      for (level = 0; level < transition_level; level++)
      {
         // Broadcast the number of nodes
         HYPRE_Int num_nodes = 0;
         if (myid == proc)
         {
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               if (hypre_ParCompGridARowPtr(compGrid[level])[i+1] - hypre_ParCompGridARowPtr(compGrid[level])[i] > 0) 
                  num_nodes++;
            }
         }
         hypre_MPI_Bcast(&num_nodes, 1, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the composite residual
         HYPRE_Complex *comp_res = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
         if (myid == proc)
         {
            HYPRE_Int cnt = 0;
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               if (hypre_ParCompGridARowPtr(compGrid[level])[i+1] - hypre_ParCompGridARowPtr(compGrid[level])[i] > 0) 
                  comp_res[cnt++] = hypre_ParCompGridF(compGrid[level])[i];
            }
         }
         hypre_MPI_Bcast(comp_res, num_nodes, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Broadcast the global indices
         HYPRE_Int *global_indices = hypre_CTAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
         if (myid == proc)
         {
            HYPRE_Int cnt = 0;
            for (i = 0; i < hypre_ParCompGridNumNodes(compGrid[level]); i++)
            {
               if (hypre_ParCompGridARowPtr(compGrid[level])[i+1] - hypre_ParCompGridARowPtr(compGrid[level])[i] > 0) 
                  global_indices[cnt++] = hypre_ParCompGridGlobalIndices(compGrid[level])[i];
            }
         }
         hypre_MPI_Bcast(global_indices, num_nodes, HYPRE_MPI_INT, proc, hypre_MPI_COMM_WORLD);

         // Now, each processors checks their owned residual value against the composite residual
         HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[level]);
         for (i = 0; i < num_nodes; i++)
         {
            if (global_indices[i] <= proc_last_index && global_indices[i] >= proc_first_index)
            {
               if (comp_res[i] != hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[level]))[global_indices[i] - proc_first_index] )
               {
                  // printf("Error: on proc %d has incorrect residual at global index %d on level %d, checked by rank %d\n", proc, global_indices[i], level, myid);
                  test_failed = 1;
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(comp_res, HYPRE_MEMORY_HOST);
            hypre_TFree(global_indices, HYPRE_MEMORY_HOST);
         }
      }
      if (transition_level != num_levels)
      {
         HYPRE_Int num_nodes = hypre_ParCompGridNumNodes(compGrid[transition_level]);

         // Broadcast the composite residual
         HYPRE_Complex *comp_res;
         if (myid == proc) comp_res = hypre_ParCompGridF(compGrid[transition_level]);
         else comp_res = hypre_CTAlloc(HYPRE_Complex, num_nodes, HYPRE_MEMORY_HOST);
         hypre_MPI_Bcast(comp_res, num_nodes, HYPRE_MPI_COMPLEX, proc, hypre_MPI_COMM_WORLD);

         // Now, each processors checks their owned residual value against the composite residual
         HYPRE_Int proc_first_index = hypre_ParVectorFirstIndex(hypre_ParAMGDataUArray(amg_data)[transition_level]);
         HYPRE_Int proc_last_index = hypre_ParVectorLastIndex(hypre_ParAMGDataUArray(amg_data)[transition_level]);
         for (i = 0; i < num_nodes; i++)
         {
            if (i <= proc_last_index && i >= proc_first_index)
            {
               if (comp_res[i] != hypre_VectorData(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[transition_level]))[i - proc_first_index] )
               {
                  // printf("Error: on proc %d has incorrect residual at global index %d on transition_level %d, checked by rank %d\n", proc, i, transition_level, myid);
                  test_failed = 1;
               }
            }
         }

         // Clean up memory
         if (myid != proc) 
         {
            hypre_TFree(comp_res, HYPRE_MEMORY_HOST);
         }         
      }
   }

   return test_failed;
}

HYPRE_Int
AgglomeratedProcessorsLocalResidualAllgather(hypre_ParAMGData *amg_data)
{
   hypre_ParCompGrid **compGrid = hypre_ParAMGDataCompGrid(amg_data);
   hypre_ParCompGridCommPkg *compGridCommPkg = hypre_ParAMGDataCompGridCommPkg(amg_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int transition_level = hypre_ParCompGridCommPkgTransitionLevel(compGridCommPkg);
   if (transition_level < 0) transition_level = num_levels;
   HYPRE_Int level, i, j, proc;

   for (level = 0; level < transition_level; level++)
   {
      // If a local communicator is stored on this level
      if (hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level]) 
      {
         // Get comm info
         MPI_Comm local_comm = hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[level];
         HYPRE_Int local_myid, local_num_procs;
         hypre_MPI_Comm_rank(local_comm, &local_myid);
         hypre_MPI_Comm_size(local_comm, &local_num_procs);

         // Count and pack up owned residual values from this level down
         HYPRE_Int *recvcounts = hypre_CTAlloc(HYPRE_Int, local_num_procs, HYPRE_MEMORY_HOST);
         for (i = level; i < transition_level; i++)
         {
            if (i > level && hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[i]) break;
            for (j = 0; j < local_num_procs; j++)
            {
               recvcounts[j] += hypre_ParCompGridOwnedBlockStarts(compGrid[i])[j+1] - hypre_ParCompGridOwnedBlockStarts(compGrid[i])[j];
            }
         }
         HYPRE_Int *displs = hypre_CTAlloc(HYPRE_Int, local_num_procs, HYPRE_MEMORY_HOST);
         for (i = 1; i < local_num_procs; i++) displs[i] = displs[i-1] + recvcounts[i-1];
         HYPRE_Complex *sendbuf = hypre_CTAlloc(HYPRE_Complex, recvcounts[local_myid], HYPRE_MEMORY_HOST);
         HYPRE_Int cnt = 0;
         for (i = level; i < transition_level; i++)
         {
            if (i > level && hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[i]) break;
            HYPRE_Int start = hypre_ParCompGridOwnedBlockStarts(compGrid[i])[local_myid];
            HYPRE_Int finish = hypre_ParCompGridOwnedBlockStarts(compGrid[i])[local_myid+1];
            for (j = start; j < finish; j++) sendbuf[cnt++] = hypre_ParCompGridF(compGrid[i])[j];
         }

         // Do the allgather
         HYPRE_Complex *recvbuf = hypre_CTAlloc(HYPRE_Complex, displs[local_num_procs-1] + recvcounts[local_num_procs-1], HYPRE_MEMORY_HOST);
         hypre_MPI_Allgatherv(sendbuf, recvcounts[local_myid], HYPRE_MPI_COMPLEX, recvbuf, recvcounts, displs, HYPRE_MPI_COMPLEX, local_comm);

         // Unpack values into comp grid
         cnt = 0;
         for (proc = 0; proc < local_num_procs; proc++)
         {
            for (i = level; i < transition_level; i++)
            {
               if (i > level && hypre_ParCompGridCommPkgAggLocalComms(compGridCommPkg)[i]) break;
               HYPRE_Int start = hypre_ParCompGridOwnedBlockStarts(compGrid[i])[proc];
               HYPRE_Int finish = hypre_ParCompGridOwnedBlockStarts(compGrid[i])[proc+1];
               for (j = start; j < finish; j++) hypre_ParCompGridF(compGrid[i])[j] = recvbuf[cnt++];
            }
         }
      }
   }

   return 0;
}


HYPRE_Int
MyZFPCompress(hypre_ParAMGData *amg_data, HYPRE_Complex *uncompressed_buffer, HYPRE_Int uncompressed_buffer_size, void **compressed_buffer, HYPRE_Int compressed_buffer_size, HYPRE_Int decompress, HYPRE_Real *zfp_errors)
{
   // Get zfp parameters
   HYPRE_Int zfp_mode = hypre_ParAMGDataUseZFPCompression(amg_data);
   double rate = hypre_ParAMGDataZFPRate(amg_data);
   double precision = hypre_ParAMGDataZFPPrecision(amg_data);
   double accuracy = hypre_ParAMGDataZFPAccuracy(amg_data);

   // Declare zfp structs
   zfp_field *field;
   zfp_stream *zfp;
   bitstream *stream;

   // Associate a zfp_field with the uncompressed send buffer
   field = zfp_field_1d(uncompressed_buffer, zfp_type_double, uncompressed_buffer_size); 
   
   // Setup ZFP stream parameters and mode
   zfp = zfp_stream_open(NULL);
   if (zfp_mode == 1) zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
   if (zfp_mode == 2) zfp_stream_set_precision(zfp, precision);
   if (zfp_mode == 3) zfp_stream_set_accuracy(zfp, accuracy);

   // Get size of compressed buffer, allocate, and attach bistream to compressed buffer, and attach zfp stream to bitstream
   if (!decompress)
   {
      compressed_buffer_size = zfp_stream_maximum_size(zfp, field);
      (*compressed_buffer) = malloc(compressed_buffer_size);
   }
   stream = stream_open((*compressed_buffer), compressed_buffer_size);
   zfp_stream_set_bit_stream(zfp, stream);
   zfp_stream_rewind(zfp);

   // Do the compression or decompression
   size_t zfpsize = 0;
   if (!decompress)
   {
      zfpsize = zfp_compress(zfp, field);
      if (!zfpsize) printf("Compression failed!\n");
      // printf("uncompressed_buffer_size = %d, compressed_buffer_size = %d, zfpsize = %d\n", uncompressed_buffer_size*8, compressed_buffer_size, zfpsize);
   }
   else
   {
      if (!zfp_decompress(zfp, field)) printf("Decompression failed!\n");
   }

   // Measure the component-wise and block relative errors
   if (!decompress && zfp_errors)
   {
      // Close the field and streams
      zfp_field_free(field);
      zfp_stream_close(zfp);

      // Reset field and streams and decompress buffer
      HYPRE_Complex *decompressed_buffer = hypre_CTAlloc(HYPRE_Complex, uncompressed_buffer_size, HYPRE_MEMORY_HOST);
      field = zfp_field_1d(decompressed_buffer, zfp_type_double, uncompressed_buffer_size);
      zfp = zfp_stream_open(NULL);
      if (zfp_mode == 1) zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
      if (zfp_mode == 2) zfp_stream_set_precision(zfp, precision);
      if (zfp_mode == 3) zfp_stream_set_accuracy(zfp, accuracy);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);
      zfp_decompress(zfp, field);

      // Compare uncompressed and decompressed buffers to get component-wise and block relative errors
      HYPRE_Real component_wise_error = 0;
      HYPRE_Real block_error = 0;
      HYPRE_Real x_inf_norm = 0;
      HYPRE_Int i;
      for (i = 0; i < uncompressed_buffer_size; i++)
      {
         if (fabs(uncompressed_buffer[i]) > x_inf_norm) x_inf_norm = fabs(uncompressed_buffer[i]);
         if (fabs(uncompressed_buffer[i] - decompressed_buffer[i]) > block_error) block_error = fabs(uncompressed_buffer[i] - decompressed_buffer[i]);
         if (uncompressed_buffer[i] != 0.0) if (fabs((uncompressed_buffer[i] - decompressed_buffer[i]) / uncompressed_buffer[i]) > component_wise_error) component_wise_error = fabs((uncompressed_buffer[i] - decompressed_buffer[i]) / uncompressed_buffer[i]);
      }
      zfp_errors[0] = component_wise_error;
      zfp_errors[1] = block_error / x_inf_norm;

      // Close the field and streams
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(stream);

      // Get the variation in the exponent for the uncompressed data
      int e_min = 999, e_max = -999;
      int e;
      int i_min = 0, i_max = 0;
      for (i = 1; i < uncompressed_buffer_size; i++)
      {
         if (uncompressed_buffer[i] != 0.0)
         {
            frexp(uncompressed_buffer[i], &e);
            if (e < e_min)
            {
               i_min = i;
               e_min = e;
            }
            if (e > e_max)
            {
               i_max = i;
               e_max = e;
            }
         }
      }
      // printf("max = %e, e = %d, min = %e, e = %d\n", uncompressed_buffer[i_max], e_max, uncompressed_buffer[i_min], e_min);
      zfp_errors[2] = e_max - e_min;
   }
   else
   {
      // Close the field and streams
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(stream);
   }

   if (zfp_mode == 1) return compressed_buffer_size;
   else return (HYPRE_Int) zfpsize;
}

HYPRE_Int
GetZFPFixedRateCompressedSizes(double rate, HYPRE_Complex *uncompressed_buffer, HYPRE_Int uncompressed_buffer_size)
{
   // Declare zfp structs
   zfp_field *field;
   zfp_stream *zfp;

   // Associate a zfp_field with the uncompressed send buffer
   field = zfp_field_1d(uncompressed_buffer, zfp_type_double, uncompressed_buffer_size); 
   
   // Setup ZFP stream parameters and mode
   zfp = zfp_stream_open(NULL);
   zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);

   // Get size of compressed buffer
   HYPRE_Int compressed_buffer_size = zfp_stream_maximum_size(zfp, field);

   // Close the field and stream
   zfp_field_free(field);
   zfp_stream_close(zfp);

   return compressed_buffer_size;
}