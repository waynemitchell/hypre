// Macros common to both coarsening and interpolation:

// The following macros all depend on the _i and _j matrices being
// defined in the scope in which they are invoked.

#define foreach_nonzero(matrix, row, column, offset)\
   for (offset=matrix##_i[row], column=matrix##_j[offset];\
        offset < matrix##_i[row+1]; offset++, column=matrix##_j[offset])

#define foreach_nonzero_skipfirst(matrix, row, column, offset)\
   for (offset=matrix##_i[row]+1, column=matrix##_j[offset];\
        offset < matrix##_i[row+1]; offset++, column=matrix##_j[offset])

#define foreach_nonzero_mapped(matrix, row, column, offset, map)\
   for (offset=matrix##_i[row], column = map ? map[matrix##_j[offset]] : \
        matrix##_j[offset]; offset < matrix##_i[row+1];\
        offset++, column = map ? map[matrix##_j[offset]] : matrix##_j[offset])


// Routines common to most interpolation types:

// Ideally would pass in P instead of P_diag_i,
// but it is not allocated when this routine is called.
void SetNonZeroOffset(HYPRE_Int *P_diag_i, HYPRE_Int node, HYPRE_Int offset);

HYPRE_Int GetNonZeroOffset(HYPRE_Int *P_diag_i, HYPRE_Int node);

void AccumulateNonZeroOffset(HYPRE_Int *P_diag_i, HYPRE_Int node, HYPRE_Int offset);

void OnCPointSetAndInc(HYPRE_Int *P_diag_i, HYPRE_Int *P_marker,
                       HYPRE_Int *jj_counter, HYPRE_Int node, HYPRE_Int node1);

void OnCPointSetAndInc_WithCF(HYPRE_Int *P_offd_i, HYPRE_Int *P_marker_offd,
                              HYPRE_Int *tmp_CF_marker_offd,
                              HYPRE_Int *jj_counter_offd,
                              HYPRE_Int node, HYPRE_Int node1);

void InterpSetIdentNoMarker(HYPRE_Int *P_j, HYPRE_Real *P_data,
                            HYPRE_Int node, HYPRE_Int *fine_to_coarse,
                            HYPRE_Int *offset);

void InterpSetIdentNoMarkerNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                  HYPRE_Int node, HYPRE_Int *offset);

void InterpSetIdentSetOffset(HYPRE_Int *P_i, HYPRE_Int *P_j,
                             HYPRE_Real *P_data, HYPRE_Int node,
                             HYPRE_Int *fine_to_coarse,
                             HYPRE_Int *offset);

void InterpSetZero(HYPRE_Int *P_j, HYPRE_Real *P_data,
                   HYPRE_Int *P_marker, HYPRE_Int node,
                   HYPRE_Int *fine_to_coarse,
                   HYPRE_Int *offset);

void InterpSetZeroNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                         HYPRE_Int *P_marker,
                         HYPRE_Int node, HYPRE_Int *offset);

void InitDiagonal(hypre_CSRMatrix *A_csr, HYPRE_Int node, HYPRE_Real *diagonal);

void InitSgn(hypre_CSRMatrix *A_csr, HYPRE_Int node, HYPRE_Int *sgn);

// NOTE:  The offset value is meant to be the one that
// is produced within the foreach_nonzero macros.
// If using a non-CSR matrix, there may not be an offset per se,
// but whatever appropriate index is used to dereference a value within
// that matrix format can be used instead, in conjunction between the macros
// and this function.
HYPRE_Int GetColumn(hypre_CSRMatrix *A_csr, HYPRE_Int offset);

void AccumulateConnection(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                          HYPRE_Real *sum);

void AccumulateIntoP(hypre_CSRMatrix *A_csr, HYPRE_Real *P_csr_data,
                     HYPRE_Int *P_marker, HYPRE_Int column, HYPRE_Int offset);

// The code is nearly identical to AccumulateConnection,
// but keep separate for clarity.
void DistributeConnection(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                          HYPRE_Real distribute, HYPRE_Real *sum);

void DistributeIntoP(hypre_CSRMatrix *A_csr, HYPRE_Real *P_csr_data,
                     HYPRE_Int *P_marker, HYPRE_Int column, HYPRE_Int offset,
                     HYPRE_Real distribute);

HYPRE_Int IsCorrectSign(hypre_CSRMatrix *A_csr, HYPRE_Int sgn,
                      HYPRE_Int offset);

void InitDistribute(hypre_CSRMatrix *A_csr, HYPRE_Int offset,
                    HYPRE_Real sum, HYPRE_Real *distribute);

void ScaleRowByDiagonal(HYPRE_Int begin_row, HYPRE_Int end_row,
                        HYPRE_Real *matrix_csr_diag_data,
                        HYPRE_Real diagonal);

void CompressP(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *P,
               hypre_CSRMatrix *P_diag,  HYPRE_Int *P_diag_i,
               HYPRE_Int *P_diag_j, HYPRE_Real *P_diag_data,
               HYPRE_Int P_diag_size,
               hypre_CSRMatrix *P_offd, HYPRE_Int *P_offd_i,
               HYPRE_Int *P_offd_j, HYPRE_Real *P_offd_data,
               HYPRE_Int P_offd_size,
               HYPRE_Real trunc_factor, HYPRE_Int max_elmts,
               HYPRE_Int *CF_marker, HYPRE_Int num_cols_A_offd,
               HYPRE_Int n_fine, HYPRE_Int *fine_to_coarse_offd);


void InitAhatStdInterp(hypre_CSRMatrix *A_csr, HYPRE_Int node,
                       HYPRE_Real *ahat, HYPRE_Int *cnt_f);

void AccumulateConnectionStdInterp(hypre_CSRMatrix *A_csr, HYPRE_Int *ihat,
                                   HYPRE_Int *ipnt, HYPRE_Real *ahat,
                                   HYPRE_Int *cnt_c, HYPRE_Int *cnt_f,
                                   HYPRE_Int *P_marker, HYPRE_Int *CF_marker,
                                   HYPRE_Int begin_row, HYPRE_Int column,
                                   HYPRE_Int offset);

void DistributeConnectionStdInterp(hypre_CSRMatrix *A_csr, HYPRE_Int *ihat,
                                   HYPRE_Int *ipnt, HYPRE_Real *ahat,
                                   HYPRE_Int *cnt_c, HYPRE_Int *cnt_f,
                                   HYPRE_Int *P_marker, HYPRE_Int begin_row,
                                   HYPRE_Int column, HYPRE_Int offset,
                                   HYPRE_Real distribute);

void InterpolateWeightStdInterp(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                HYPRE_Int *ihat, HYPRE_Int *ipnt, HYPRE_Real *ahat,
                                HYPRE_Real alfa, HYPRE_Real beta,
                                HYPRE_Int begin_row, HYPRE_Int end_row,
                                HYPRE_Real cnt_f, HYPRE_Int *fine_to_coarse);

void InterpolateWeightStdInterpNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                      HYPRE_Int *ihat, HYPRE_Int *ipnt,
                                      HYPRE_Real *ahat,
                                      HYPRE_Real alfa, HYPRE_Real beta,
                                      HYPRE_Int begin_row, HYPRE_Int end_row,
                                      HYPRE_Real cnt_f);

void InterpolateWeightStdInterpNoCheck(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                       HYPRE_Int *ihat, HYPRE_Int *ipnt,
                                       HYPRE_Real *ahat,
                                       HYPRE_Real alfa,
                                       HYPRE_Int begin_row, HYPRE_Int end_row,
                                       HYPRE_Real cnt_f,
                                       HYPRE_Int *fine_to_coarse);

void InterpolateWeightStdInterpNoCheckNoFToC(HYPRE_Int *P_j, HYPRE_Real *P_data,
                                             HYPRE_Int *ihat, HYPRE_Int *ipnt,
                                             HYPRE_Real *ahat, HYPRE_Real alfa,
                                             HYPRE_Int begin_row, HYPRE_Int end_row,
                                             HYPRE_Real cnt_f);
