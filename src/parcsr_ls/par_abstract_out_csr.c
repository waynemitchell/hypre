#include "_hypre_parcsr_ls.h"

// Ideally would pass in P instead of P_diag_i,
// but it is not allocated when this routine is called.
void SetNonZeroOffset(HYPRE_Int *A_csr_i, HYPRE_Int node, HYPRE_Int offset)
{
   A_csr_i[node] = offset;
}

HYPRE_Int GetNonZeroOffset(HYPRE_Int *A_csr_i, HYPRE_Int node)
{
   return A_csr_i[node];
}

void AccumulateNonZeroOffset(HYPRE_Int *A_csr_i, HYPRE_Int node, HYPRE_Int offset)
{
   A_csr_i[node] += offset;
}

void OnCPointSetAndInc(HYPRE_Int *P_diag_i, HYPRE_Int *P_marker,
                       HYPRE_Int *jj_counter, HYPRE_Int node, HYPRE_Int node1)
{
   if (P_marker[node1] < P_diag_i[node])
   {
      P_marker[node1] = *jj_counter;
      (*jj_counter)++;
   }
}

void OnCPointSetAndInc_WithCF(HYPRE_Int *P_offd_i, HYPRE_Int *P_marker_offd,
                              HYPRE_Int *tmp_CF_marker_offd,
                              HYPRE_Int *jj_counter_offd,
                              HYPRE_Int node, HYPRE_Int node1)
{
   if(P_marker_offd[node1] < P_offd_i[node])
   {
      tmp_CF_marker_offd[node1] = 1;
      P_marker_offd[node1] = *jj_counter_offd;
      (*jj_counter_offd)++;
   }
}

void InterpSetIdentNoMarker(HYPRE_Int *P_j, HYPRE_Real *P_data,
                            HYPRE_Int node, HYPRE_Int *fine_to_coarse,
                            HYPRE_Int *offset)
{
   P_j[*offset] = fine_to_coarse[node];
   P_data[*offset] = 1.0;
   (*offset)++;
}

void InterpSetIdentNoMarkerNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                  HYPRE_Int node, HYPRE_Int *offset)
{
   P_j[*offset] = node;
   P_data[*offset] = 1.0;
   (*offset)++;
}

void InterpSetIdentSetOffset(HYPRE_Int *P_i, HYPRE_Int *P_j,
                             HYPRE_Real *P_data, HYPRE_Int node,
                             HYPRE_Int *fine_to_coarse,
                             HYPRE_Int *offset)
{
   P_i[node] = *offset;
   P_j[*offset] = fine_to_coarse[node];
   P_data[*offset] = 1.0;
   (*offset)++;
}

void InterpSetZero(HYPRE_Int *P_j, HYPRE_Real *P_data,
                   HYPRE_Int *P_marker, HYPRE_Int node,
                   HYPRE_Int *fine_to_coarse,
                   HYPRE_Int *offset)
{
   P_marker[node] = *offset;
   P_j[*offset] = fine_to_coarse[node];
   P_data[*offset] = 0.0;
   (*offset)++;
}

void InterpSetZeroNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                         HYPRE_Int *P_marker,
                         HYPRE_Int node, HYPRE_Int *offset)
{
   P_marker[node] = *offset;
   P_j[*offset] = node;
   P_data[*offset] = 0.0;
   (*offset)++;
}

void InitDiagonal(hypre_CSRMatrix *A_csr, HYPRE_Int node, HYPRE_Real *diagonal)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);
   HYPRE_Int  *A_csr_i = hypre_CSRMatrixI(A_csr);

   *diagonal = A_csr_data[A_csr_i[node]];
}

void InitSgn(hypre_CSRMatrix *A_csr, HYPRE_Int node, HYPRE_Int *sgn)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);
   HYPRE_Int  *A_csr_i = hypre_CSRMatrixI(A_csr);

   *sgn = 1;
   if(A_csr_data[A_csr_i[node]] < 0) *sgn = -1;
}

// NOTE:  For the remaining functions, the offset value is meant
// to be the one that is produced within the foreach_nonzero macros.
// If using a non-CSR matrix, there may not be an offset per se,
// but whatever appropriate index is used to dereference a value within
// that matrix format can be used instead, in conjunction between the macros
// and this function.

HYPRE_Int GetColumn(hypre_CSRMatrix *A_csr, HYPRE_Int offset)
{
   HYPRE_Int  *A_csr_j = hypre_CSRMatrixJ(A_csr);

   return A_csr_j[offset];
}

void AccumulateConnection(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                          HYPRE_Real *sum)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);

   *sum += A_csr_data[offset];
}

void AccumulateIntoP(hypre_CSRMatrix *A_csr, HYPRE_Real *P_csr_data,
                     HYPRE_Int *P_marker, HYPRE_Int column, HYPRE_Int offset)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);
   //HYPRE_Real *P_csr_data = hypre_CSRMatrixData(P_csr);

   P_csr_data[P_marker[column]] +=  A_csr_data[offset];
}

// The code is nearly identical to AccumulateConnection,
// but keep separate for clarity.
void DistributeConnection(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                          HYPRE_Real distribute, HYPRE_Real *sum)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);

   *sum += distribute * A_csr_data[offset];
}

void DistributeIntoP(hypre_CSRMatrix *A_csr, HYPRE_Real *P_csr_data,
                     HYPRE_Int *P_marker, HYPRE_Int column, HYPRE_Int offset,
                     HYPRE_Real distribute)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);
   //HYPRE_Real *P_csr_data = hypre_CSRMatrixData(P_csr);

   P_csr_data[P_marker[column]] +=  distribute * A_csr_data[offset];
}

HYPRE_Int IsCorrectSign(hypre_CSRMatrix *A_csr, HYPRE_Int sgn,
                      HYPRE_Int offset)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);

   return sgn * A_csr_data[offset] < 0;
}

void ScaleBySum(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                HYPRE_Real sum, HYPRE_Real *distribute)
{
   HYPRE_Real *A_csr_data = hypre_CSRMatrixData(A_csr);

   *distribute = A_csr_data[offset] / sum;
}

void ScaleRowByDiagonal(HYPRE_Int begin_row, HYPRE_Int end_row,
                        HYPRE_Real *matrix_csr_data,
                        HYPRE_Real diagonal)
{
   HYPRE_Int jj;

   for(jj = begin_row; jj < end_row; jj++)
      matrix_csr_data[jj] /= -diagonal;
}

void CompressP(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
               hypre_CSRMatrix *P_diag,  HYPRE_Int *P_diag_i,
               HYPRE_Int *P_diag_j, HYPRE_Real *P_diag_data,
               HYPRE_Int P_diag_size,
               hypre_CSRMatrix *P_offd, HYPRE_Int *P_offd_i,
               HYPRE_Int *P_offd_j, HYPRE_Real *P_offd_data,
               HYPRE_Int P_offd_size,
               HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
               HYPRE_Int *CF_marker, HYPRE_Int num_cols_A_offd,
               HYPRE_Int n_fine, HYPRE_Int *fine_to_coarse_offd)
{
   HYPRE_Int num_cols_P_offd;
   HYPRE_Int *col_map_offd_P;
   HYPRE_Int *P_marker;
   HYPRE_Int i, index;

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i=0; i < num_cols_A_offd; i++)
	 P_marker[i] = 0;

      num_cols_P_offd = 0;
      for (i=0; i < P_offd_size; i++)
      {
#if 0
         index = GetColumn(P_offd_j, i);
#endif
	 index = P_offd_j[i];

	 if (!P_marker[index])
	 {
 	    num_cols_P_offd++;
 	    P_marker[index] = 1;
  	 }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int,num_cols_P_offd);

      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         col_map_offd_P[i] = index++;
      }

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i=0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      hypre_TFree(P_marker); 
   }

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;

   if (num_cols_P_offd)
   { 
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   } 

   hypre_GetCommPkgRTFromCommPkgA(P,A, fine_to_coarse_offd);
}
