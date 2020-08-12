/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_AMGDDCompGrid and hypre_AMGDDCommPkg classes.
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"
#include <stdio.h>
#include <math.h>


HYPRE_Int hypre_AMGDDCompGridLocalIndexBinarySearch( hypre_AMGDDCompGrid *compGrid, HYPRE_Int global_index )
{
   HYPRE_Int      left = 0;
   HYPRE_Int      right = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid)-1;
   HYPRE_Int      index, sorted_index;
   HYPRE_Int      *inv_map = hypre_AMGDDCompGridNonOwnedInvSort(compGrid);

   while (left <= right)
   {
      sorted_index = (left + right) / 2;
      index = inv_map[sorted_index];
      if (hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[index] < global_index) left = sorted_index + 1;
      else if (hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[index] > global_index) right = sorted_index - 1;
      else return index;
   }

   return -1;
}

hypre_AMGDDCompGridMatrix* hypre_AMGDDCompGridMatrixCreate()
{
   hypre_AMGDDCompGridMatrix *matrix = hypre_CTAlloc(hypre_AMGDDCompGridMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_AMGDDCompGridMatrixOwnedDiag(matrix) = NULL;
   hypre_AMGDDCompGridMatrixOwnedOffd(matrix) = NULL;
   hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix) = NULL;
   hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix) = NULL;

   hypre_AMGDDCompGridMatrixRealReal(matrix) = NULL;
   hypre_AMGDDCompGridMatrixRealGhost(matrix) = NULL;

   hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix) = 0;
   hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix) = 0;

   return matrix;
}

HYPRE_Int hypre_AMGDDCompGridMatrixDestroy(hypre_AMGDDCompGridMatrix *matrix)
{
   if (hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(matrix))
   {
      if (hypre_AMGDDCompGridMatrixOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixOwnedDiag(matrix));
      if (hypre_AMGDDCompGridMatrixOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixOwnedOffd(matrix));
   }
   else if (hypre_AMGDDCompGridMatrixOwnsOffdColIndices(matrix))
   {
      if (hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(matrix))) hypre_TFree(hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(matrix)), hypre_CSRMatrixMemoryLocation(hypre_AMGDDCompGridMatrixOwnedOffd(matrix)));
      if (hypre_AMGDDCompGridMatrixOwnedOffd(matrix)) hypre_TFree(hypre_AMGDDCompGridMatrixOwnedOffd(matrix), HYPRE_MEMORY_HOST);
   }
   if (hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixNonOwnedDiag(matrix));
   if (hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixNonOwnedOffd(matrix));
   if (hypre_AMGDDCompGridMatrixRealReal(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixRealReal(matrix));
   if (hypre_AMGDDCompGridMatrixRealGhost(matrix)) hypre_CSRMatrixDestroy(hypre_AMGDDCompGridMatrixRealGhost(matrix));

   hypre_TFree(matrix, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridMatrixSetupRealMatvec(hypre_AMGDDCompGridMatrix *A)
{
   HYPRE_Int i,j;

   hypre_CSRMatrix *A_real_real = hypre_AMGDDCompGridMatrixRealReal(A);
   hypre_CSRMatrixInitialize(A_real_real);

   hypre_CSRMatrix *A_real_ghost = hypre_AMGDDCompGridMatrixRealGhost(A);
   hypre_CSRMatrixInitialize(A_real_ghost);

   hypre_CSRMatrix *A_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);

   HYPRE_Int num_real = hypre_CSRMatrixNumRows(A_real_real);
   HYPRE_Int A_real_real_nnz = 0;
   HYPRE_Int A_real_ghost_nnz = 0;

   for (i = 0; i < num_real; i++)
   {
      hypre_CSRMatrixI(A_real_real)[i] = A_real_real_nnz;
      hypre_CSRMatrixI(A_real_ghost)[i] = A_real_ghost_nnz;
      for (j = hypre_CSRMatrixI(A_diag)[i]; j < hypre_CSRMatrixI(A_diag)[i+1]; j++)
      {
         HYPRE_Int col_ind = hypre_CSRMatrixJ(A_diag)[j];
         if (col_ind < num_real)
         {
             hypre_CSRMatrixJ(A_real_real)[A_real_real_nnz] = col_ind;
             hypre_CSRMatrixData(A_real_real)[A_real_real_nnz] = hypre_CSRMatrixData(A_diag)[j];
             A_real_real_nnz++;
         }
         else
         {
             hypre_CSRMatrixJ(A_real_ghost)[A_real_ghost_nnz] = col_ind;
             hypre_CSRMatrixData(A_real_ghost)[A_real_ghost_nnz] = hypre_CSRMatrixData(A_diag)[j];
             A_real_ghost_nnz++;
         }
      }
   }

   hypre_CSRMatrixI(A_real_real)[num_real] = A_real_real_nnz;
   hypre_CSRMatrixI(A_real_ghost)[num_real] = A_real_ghost_nnz;

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridMatvec( HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A, hypre_AMGDDCompGridVector *x, HYPRE_Complex beta, hypre_AMGDDCompGridVector *y)
{
   hypre_CSRMatrix *owned_diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   hypre_Vector *x_owned = hypre_AMGDDCompGridVectorOwned(x);
   hypre_Vector *x_nonowned = hypre_AMGDDCompGridVectorNonOwned(x);

   hypre_Vector *y_owned = hypre_AMGDDCompGridVectorOwned(y);
   hypre_Vector *y_nonowned = hypre_AMGDDCompGridVectorNonOwned(y);

   hypre_CSRMatrixMatvec(alpha, owned_diag, x_owned, beta, y_owned);
   if (owned_offd)
       hypre_CSRMatrixMatvec(alpha, owned_offd, x_nonowned, 1.0, y_owned);
   if (nonowned_diag)
       hypre_CSRMatrixMatvec(alpha, nonowned_diag, x_nonowned, beta, y_nonowned);
   if(nonowned_offd)
       hypre_CSRMatrixMatvec(alpha, nonowned_offd, x_owned, 1.0, y_nonowned);

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridRealMatvec( HYPRE_Complex alpha, hypre_AMGDDCompGridMatrix *A, hypre_AMGDDCompGridVector *x, HYPRE_Complex beta, hypre_AMGDDCompGridVector *y)
{
   if ( !hypre_CSRMatrixData( hypre_AMGDDCompGridMatrixRealReal(A) ) )
   {
      hypre_AMGDDCompGridMatrixSetupRealMatvec(A);
   }

   hypre_CSRMatrix *owned_diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixRealReal(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   hypre_Vector *x_owned = hypre_AMGDDCompGridVectorOwned(x);
   hypre_Vector *x_nonowned = hypre_AMGDDCompGridVectorNonOwned(x);

   hypre_Vector *y_owned = hypre_AMGDDCompGridVectorOwned(y);
   hypre_Vector *y_nonowned = hypre_AMGDDCompGridVectorNonOwned(y);

   hypre_CSRMatrixMatvec(alpha, owned_diag, x_owned, beta, y_owned);
   if (owned_offd)
       hypre_CSRMatrixMatvec(alpha, owned_offd, x_nonowned, 1.0, y_owned);
   if (nonowned_diag)
       hypre_CSRMatrixMatvec(alpha, nonowned_diag, x_nonowned, beta, y_nonowned);
   if(nonowned_offd)
       hypre_CSRMatrixMatvec(alpha, nonowned_offd, x_owned, 1.0, y_nonowned);

   return 0;
}

hypre_AMGDDCompGridVector *hypre_AMGDDCompGridVectorCreate()
{
   hypre_AMGDDCompGridVector *vector = hypre_CTAlloc(hypre_AMGDDCompGridVector, 1, HYPRE_MEMORY_HOST);

   hypre_AMGDDCompGridVectorOwned(vector) = NULL;
   hypre_AMGDDCompGridVectorNonOwned(vector) = NULL;

   hypre_AMGDDCompGridVectorOwnsOwnedVector(vector) = 0;

   return vector;
}

HYPRE_Int hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridVector *vector, HYPRE_Int num_owned, HYPRE_Int num_nonowned, HYPRE_Int num_real)
{
   hypre_AMGDDCompGridVectorOwned(vector) = hypre_SeqVectorCreate(num_owned);
   hypre_SeqVectorInitialize(hypre_AMGDDCompGridVectorOwned(vector));
   hypre_AMGDDCompGridVectorOwnsOwnedVector(vector) = 1;
   hypre_AMGDDCompGridVectorNumReal(vector) = num_real;
   hypre_AMGDDCompGridVectorNonOwned(vector) = hypre_SeqVectorCreate(num_nonowned);
   hypre_SeqVectorInitialize(hypre_AMGDDCompGridVectorNonOwned(vector));

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridVector *vector)
{
   if (hypre_AMGDDCompGridVectorOwnsOwnedVector(vector))
   {
      if (hypre_AMGDDCompGridVectorOwned(vector)) hypre_SeqVectorDestroy(hypre_AMGDDCompGridVectorOwned(vector));
   }
   if (hypre_AMGDDCompGridVectorNonOwned(vector)) hypre_SeqVectorDestroy(hypre_AMGDDCompGridVectorNonOwned(vector));

   hypre_TFree(vector, HYPRE_MEMORY_HOST);

   return 0;
}

HYPRE_Real hypre_AMGDDCompGridVectorInnerProd(hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y)
{
    return ( hypre_SeqVectorInnerProd(hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y))
             + hypre_SeqVectorInnerProd(hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y)) );
}

HYPRE_Real hypre_AMGDDCompGridVectorRealInnerProd(hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y)
{
    HYPRE_Int orig_x_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x));
    HYPRE_Int orig_y_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = hypre_AMGDDCompGridVectorNumReal(x);
    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = hypre_AMGDDCompGridVectorNumReal(y);

    HYPRE_Real i_prod = hypre_SeqVectorInnerProd(hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y))
             + hypre_SeqVectorInnerProd(hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;
    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = orig_y_size;

    return i_prod;
}

HYPRE_Int hypre_AMGDDCompGridVectorScale(HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x)
{
    hypre_SeqVectorScale(alpha, hypre_AMGDDCompGridVectorOwned(x));
    hypre_SeqVectorScale(alpha, hypre_AMGDDCompGridVectorNonOwned(x));

    return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorRealScale(HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x)
{
    HYPRE_Int orig_x_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = hypre_AMGDDCompGridVectorNumReal(x);

    hypre_SeqVectorScale(alpha, hypre_AMGDDCompGridVectorOwned(x));
    hypre_SeqVectorScale(alpha, hypre_AMGDDCompGridVectorNonOwned(x));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;

    return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorAxpy(HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y )
{
   if (hypre_AMGDDCompGridVectorOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y));
   if (hypre_AMGDDCompGridVectorNonOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y));

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorRealAxpy(HYPRE_Complex alpha, hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y )
{
    HYPRE_Int orig_x_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x));
    HYPRE_Int orig_y_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = hypre_AMGDDCompGridVectorNumReal(x);
    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = hypre_AMGDDCompGridVectorNumReal(y);

   if (hypre_AMGDDCompGridVectorOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y));
   if (hypre_AMGDDCompGridVectorNonOwned(x))
      hypre_SeqVectorAxpy(alpha, hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;
    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = orig_y_size;

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorSetConstantValues(hypre_AMGDDCompGridVector *vector, HYPRE_Complex value )
{
   if (hypre_AMGDDCompGridVectorOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_AMGDDCompGridVectorOwned(vector), value);
   if (hypre_AMGDDCompGridVectorNonOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_AMGDDCompGridVectorNonOwned(vector), value);

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorRealSetConstantValues(hypre_AMGDDCompGridVector *vector, HYPRE_Complex value )
{
   HYPRE_Int orig_vec_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(vector));

   hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(vector)) = hypre_AMGDDCompGridVectorNumReal(vector);

   if (hypre_AMGDDCompGridVectorOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_AMGDDCompGridVectorOwned(vector), value);
   if (hypre_AMGDDCompGridVectorNonOwned(vector))
      hypre_SeqVectorSetConstantValues(hypre_AMGDDCompGridVectorNonOwned(vector), value);

   hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(vector)) = orig_vec_size;

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorCopy(hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y )
{
   if (hypre_AMGDDCompGridVectorOwned(x) && hypre_AMGDDCompGridVectorOwned(y))
      hypre_SeqVectorCopy(hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y));
   if (hypre_AMGDDCompGridVectorNonOwned(x) && hypre_AMGDDCompGridVectorNonOwned(y))
      hypre_SeqVectorCopy(hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y));
   return 0;
}

HYPRE_Int hypre_AMGDDCompGridVectorRealCopy(hypre_AMGDDCompGridVector *x, hypre_AMGDDCompGridVector *y )
{
    HYPRE_Int orig_x_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x));
    HYPRE_Int orig_y_size = hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y));

    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = hypre_AMGDDCompGridVectorNumReal(x);
    hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = hypre_AMGDDCompGridVectorNumReal(y);

   if (hypre_AMGDDCompGridVectorOwned(x) && hypre_AMGDDCompGridVectorOwned(y))
      hypre_SeqVectorCopy(hypre_AMGDDCompGridVectorOwned(x), hypre_AMGDDCompGridVectorOwned(y));
   if (hypre_AMGDDCompGridVectorNonOwned(x) && hypre_AMGDDCompGridVectorNonOwned(y))
      hypre_SeqVectorCopy(hypre_AMGDDCompGridVectorNonOwned(x), hypre_AMGDDCompGridVectorNonOwned(y));

   hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(x)) = orig_x_size;
   hypre_VectorSize(hypre_AMGDDCompGridVectorNonOwned(y)) = orig_y_size;

   return 0;
}

hypre_AMGDDCompGrid *hypre_AMGDDCompGridCreate ()
{
   hypre_AMGDDCompGrid      *compGrid;

   compGrid = hypre_CTAlloc(hypre_AMGDDCompGrid, 1, HYPRE_MEMORY_HOST);
   hypre_AMGDDCompGridMemoryLocation(compGrid) = HYPRE_MEMORY_UNDEFINED;

   hypre_AMGDDCompGridFirstGlobalIndex(compGrid) = 0;
   hypre_AMGDDCompGridLastGlobalIndex(compGrid) = 0;
   hypre_AMGDDCompGridNumOwnedNodes(compGrid) = 0;
   hypre_AMGDDCompGridNumNonOwnedNodes(compGrid) = 0;
   hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) = 0;
   hypre_AMGDDCompGridNumOwnedCPoints(compGrid) = 0;
   hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid) = 0;
   hypre_AMGDDCompGridNumMissingColIndices(compGrid) = 0;

   hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedRealMarker(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedSort(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedInvSort(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) = NULL;

   hypre_AMGDDCompGridOwnedCoarseIndices(compGrid) = NULL;

   hypre_AMGDDCompGridA(compGrid) = NULL;
   hypre_AMGDDCompGridP(compGrid) = NULL;
   hypre_AMGDDCompGridR(compGrid) = NULL;

   hypre_AMGDDCompGridU(compGrid) = NULL;
   hypre_AMGDDCompGridF(compGrid) = NULL;
   hypre_AMGDDCompGridT(compGrid) = NULL;
   hypre_AMGDDCompGridS(compGrid) = NULL;
   hypre_AMGDDCompGridQ(compGrid) = NULL;
   hypre_AMGDDCompGridTemp(compGrid) = NULL;
   hypre_AMGDDCompGridTemp2(compGrid) = NULL;
   hypre_AMGDDCompGridTemp3(compGrid) = NULL;

   hypre_AMGDDCompGridL1Norms(compGrid) = NULL;
   hypre_AMGDDCompGridCFMarkerArray(compGrid) = NULL;
   hypre_AMGDDCompGridOwnedCMask(compGrid) = NULL;
   hypre_AMGDDCompGridOwnedFMask(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedCMask(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedFMask(compGrid) = NULL;
   hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid) = NULL;
   hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid) = NULL;

   return compGrid;
}

HYPRE_Int hypre_AMGDDCompGridDestroy ( hypre_AMGDDCompGrid *compGrid )
{
   if (hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedRealMarker(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedRealMarker(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedSort(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedSort(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedInvSort(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedInvSort(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridOwnedCoarseIndices(compGrid))
      hypre_TFree(hypre_AMGDDCompGridOwnedCoarseIndices(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));

   if (hypre_AMGDDCompGridA(compGrid))
      hypre_AMGDDCompGridMatrixDestroy(hypre_AMGDDCompGridA(compGrid));
   if (hypre_AMGDDCompGridP(compGrid))
      hypre_AMGDDCompGridMatrixDestroy(hypre_AMGDDCompGridP(compGrid));
   if (hypre_AMGDDCompGridR(compGrid))
      hypre_AMGDDCompGridMatrixDestroy(hypre_AMGDDCompGridR(compGrid));

   if (hypre_AMGDDCompGridU(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridU(compGrid));
   if (hypre_AMGDDCompGridF(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridF(compGrid));
   if (hypre_AMGDDCompGridT(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridT(compGrid));
   if (hypre_AMGDDCompGridS(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridS(compGrid));
   if (hypre_AMGDDCompGridQ(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridQ(compGrid));
   if (hypre_AMGDDCompGridTemp(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridTemp(compGrid));
   if (hypre_AMGDDCompGridTemp2(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridTemp2(compGrid));
   if (hypre_AMGDDCompGridTemp3(compGrid))
      hypre_AMGDDCompGridVectorDestroy(hypre_AMGDDCompGridTemp3(compGrid));

   if (hypre_AMGDDCompGridL1Norms(compGrid))
      hypre_TFree(hypre_AMGDDCompGridL1Norms(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridCFMarkerArray(compGrid))
      hypre_TFree(hypre_AMGDDCompGridCFMarkerArray(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridOwnedCMask(compGrid))
      hypre_TFree(hypre_AMGDDCompGridOwnedCMask(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridOwnedFMask(compGrid))
      hypre_TFree(hypre_AMGDDCompGridOwnedFMask(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedCMask(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedCMask(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedFMask(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedFMask(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid))
      hypre_TFree(hypre_AMGDDCompGridOwnedRelaxOrdering(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));
   if (hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid))
      hypre_TFree(hypre_AMGDDCompGridNonOwnedRelaxOrdering(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));

   hypre_TFree(compGrid, HYPRE_MEMORY_HOST);


   return 0;
}

HYPRE_Int hypre_AMGDDCompGridInitialize( hypre_ParAMGDDData *amgdd_data, HYPRE_Int padding, HYPRE_Int level )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int         i,j;

   // Get info from the amg data structure
   hypre_ParAMGData *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   hypre_AMGDDCompGridLevel(compGrid) = level;
   HYPRE_Int *CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data)[level];
   hypre_CSRMatrix *A_diag_original = hypre_ParCSRMatrixDiag( hypre_ParAMGDataAArray(amg_data)[level] );
   hypre_CSRMatrix *A_offd_original = hypre_ParCSRMatrixOffd( hypre_ParAMGDataAArray(amg_data)[level] );
   hypre_AMGDDCompGridFirstGlobalIndex(compGrid) = hypre_ParVectorFirstIndex(hypre_ParAMGDataFArray(amg_data)[level]);
   hypre_AMGDDCompGridLastGlobalIndex(compGrid) = hypre_ParVectorLastIndex(hypre_ParAMGDataFArray(amg_data)[level]);
   hypre_AMGDDCompGridNumOwnedNodes(compGrid) = hypre_VectorSize(hypre_ParVectorLocalVector(hypre_ParAMGDataFArray(amg_data)[level]));
   hypre_AMGDDCompGridNumNonOwnedNodes(compGrid) = hypre_CSRMatrixNumCols(A_offd_original);
   hypre_AMGDDCompGridNumMissingColIndices(compGrid) = 0;
   hypre_AMGDDCompGridMemoryLocation(compGrid) = hypre_ParCSRMatrixMemoryLocation( hypre_ParAMGDataAArray(amg_data)[level] );

   // !!! Check on how good a guess this is for eventual size of the nononwed dofs and nnz
   HYPRE_Int max_nonowned = 2 * (padding + hypre_ParAMGDDDataNumGhostLayers(amgdd_data)) * hypre_CSRMatrixNumCols(A_offd_original);
   HYPRE_Int ave_nnz_per_row = 0;
   if (hypre_CSRMatrixNumRows(A_diag_original)) ave_nnz_per_row = (HYPRE_Int) (hypre_CSRMatrixNumNonzeros(A_diag_original) / hypre_CSRMatrixNumRows(A_diag_original));
   HYPRE_Int max_nonowned_diag_nnz = max_nonowned * ave_nnz_per_row;
   HYPRE_Int max_nonowned_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd_original);

   // Setup CompGridMatrix A
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridMatrixCreate();
   hypre_AMGDDCompGridMatrixOwnedDiag(A) = A_diag_original;
   hypre_AMGDDCompGridMatrixOwnedOffd(A) = A_offd_original;
   hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(A) = 0;
   hypre_AMGDDCompGridMatrixNonOwnedDiag(A) = hypre_CSRMatrixCreate(max_nonowned, max_nonowned, max_nonowned_diag_nnz);
   hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedDiag(A));
   hypre_AMGDDCompGridMatrixNonOwnedOffd(A) = hypre_CSRMatrixCreate(max_nonowned, hypre_AMGDDCompGridNumOwnedNodes(compGrid), max_nonowned_offd_nnz);
   hypre_CSRMatrixInitialize(hypre_AMGDDCompGridMatrixNonOwnedOffd(A));
   hypre_AMGDDCompGridA(compGrid) = A;
   hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid));

   // Setup CompGridMatrix P and R if appropriate
   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      hypre_AMGDDCompGridMatrix *P = hypre_AMGDDCompGridMatrixCreate();
      hypre_AMGDDCompGridMatrixOwnedDiag(P) = hypre_ParCSRMatrixDiag( hypre_ParAMGDataPArray(amg_data)[level] );
      // Use original rowptr and data from P, but need to use new col indices (init to global index, then setup local indices later)
      hypre_CSRMatrix *P_offd_original = hypre_ParCSRMatrixOffd( hypre_ParAMGDataPArray(amg_data)[level] );
      hypre_AMGDDCompGridMatrixOwnedOffd(P) = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(P_offd_original), hypre_CSRMatrixNumCols(P_offd_original), hypre_CSRMatrixNumNonzeros(P_offd_original));
      hypre_CSRMatrixI(hypre_AMGDDCompGridMatrixOwnedOffd(P)) = hypre_CSRMatrixI(P_offd_original);
      hypre_CSRMatrixData(hypre_AMGDDCompGridMatrixOwnedOffd(P)) = hypre_CSRMatrixData(P_offd_original);
      hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(P)) = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(P_offd_original), hypre_AMGDDCompGridMemoryLocation(compGrid));

      // Initialize P owned offd col ind to their global indices
      for (i = 0; i < hypre_CSRMatrixNumNonzeros(hypre_AMGDDCompGridMatrixOwnedOffd(P)); i++)
      {
         hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(P))[i] = hypre_ParCSRMatrixColMapOffd( hypre_ParAMGDataPArray(amg_data)[level] )[ hypre_CSRMatrixJ(P_offd_original)[i] ];
      }

      hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(P) = 0;
      hypre_AMGDDCompGridMatrixOwnsOffdColIndices(P) = 1;
      hypre_AMGDDCompGridP(compGrid) = P;

      if (hypre_ParAMGDataRestriction(amg_data))
      {
         hypre_AMGDDCompGridMatrix *R = hypre_AMGDDCompGridMatrixCreate();
         hypre_AMGDDCompGridMatrixOwnedDiag(R) = hypre_ParCSRMatrixDiag( hypre_ParAMGDataRArray(amg_data)[level] );
         // Use original rowptr and data from R, but need to use new col indices (init to global index, then setup local indices later)
         hypre_CSRMatrix *R_offd_original = hypre_ParCSRMatrixOffd( hypre_ParAMGDataRArray(amg_data)[level] );
         hypre_AMGDDCompGridMatrixOwnedOffd(R) = hypre_CSRMatrixCreate(hypre_CSRMatrixNumRows(R_offd_original), hypre_CSRMatrixNumCols(R_offd_original), hypre_CSRMatrixNumNonzeros(R_offd_original));
         hypre_CSRMatrixI(hypre_AMGDDCompGridMatrixOwnedOffd(R)) = hypre_CSRMatrixI(R_offd_original);
         hypre_CSRMatrixData(hypre_AMGDDCompGridMatrixOwnedOffd(R)) = hypre_CSRMatrixData(R_offd_original);
         hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(R)) = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumNonzeros(R_offd_original), hypre_AMGDDCompGridMemoryLocation(compGrid));

         // Initialize R owned offd col ind to their global indices
         for (i = 0; i < hypre_CSRMatrixNumNonzeros(hypre_AMGDDCompGridMatrixOwnedOffd(R)); i++)
         {
            hypre_CSRMatrixJ(hypre_AMGDDCompGridMatrixOwnedOffd(R))[i] = hypre_ParCSRMatrixColMapOffd( hypre_ParAMGDataRArray(amg_data)[level] )[ hypre_CSRMatrixJ(R_offd_original)[i] ];
         }

         hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(R) = 0;
         hypre_AMGDDCompGridMatrixOwnsOffdColIndices(R) = 1;
         hypre_AMGDDCompGridR(compGrid) = R;
      }
   }

   // Allocate some extra arrays used during AMG-DD setup
   hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid));
   hypre_AMGDDCompGridNonOwnedRealMarker(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid));
   hypre_AMGDDCompGridNonOwnedSort(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid));
   hypre_AMGDDCompGridNonOwnedInvSort(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid));

   // Initialize nonowned global indices, real marker, and the sort and invsort arrays
   for (i = 0; i < hypre_CSRMatrixNumCols(A_offd_original); i++)
   {
      hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid)[i] = hypre_ParCSRMatrixColMapOffd( hypre_ParAMGDataAArray(amg_data)[level] )[i];
      hypre_AMGDDCompGridNonOwnedSort(compGrid)[i] = i;
      hypre_AMGDDCompGridNonOwnedInvSort(compGrid)[i] = i;
      hypre_AMGDDCompGridNonOwnedRealMarker(compGrid)[i] = 1; // NOTE: Assume that padding is at least 1, i.e. first layer of points are real
   }

   if (level != hypre_ParAMGDataNumLevels(amg_data) - 1)
   {
      hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, max_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_AMGDDCompGridOwnedCoarseIndices(compGrid) = hypre_CTAlloc(HYPRE_Int, hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridMemoryLocation(compGrid));

      // Setup the owned coarse indices
      if ( CF_marker_array )
      {
         HYPRE_Int coarseIndexCounter = 0;
         for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
         {
            if ( CF_marker_array[i] == 1 )
            {
               hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = coarseIndexCounter++;
            }
            else
            {
               hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = -1;
            }
         }
      }
      else
      {
         for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
         {
            hypre_AMGDDCompGridOwnedCoarseIndices(compGrid)[i] = -1;
         }
      }
   }

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridSetupRelax( hypre_ParAMGDDData *amgdd_data )
{
   HYPRE_Int level, i, j;

   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   if (hypre_ParAMGDataAMGDDFACUsePCG(amg_data))
   {
       hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_PCG;
   }
   else
   {
       // Default to CFL1 Jacobi
       hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi; 
       if (hypre_ParAMGDataAMGDDFACRelaxType(amg_data) == 0) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_Jacobi;
       else if (hypre_ParAMGDataAMGDDFACRelaxType(amg_data) == 1) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_GaussSeidel;
       else if (hypre_ParAMGDataAMGDDFACRelaxType(amg_data) == 2) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_OrderedGaussSeidel; 
       else if (hypre_ParAMGDataAMGDDFACRelaxType(amg_data) == 3) hypre_ParAMGDataAMGDDUserFACRelaxation(amg_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi; 
       else 
       {
          hypre_printf("Warning: unknown AMGDD FAC relaxation type. Defaulting to CFL1 Jacobi.\n");
          hypre_ParAMGDataAMGDDFACRelaxType(amg_data) = 3;
       }
   }

   // Default to CFL1 Jacobi
   if (hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 0) hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_Jacobi;
   else if (hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 1) hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_GaussSeidel;
   else if (hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 2) hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_OrderedGaussSeidel;
   else if (hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 3) hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi;
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"WARNING: unknown AMGDD FAC relaxation type. Defaulting to CFL1 Jacobi.\n");
      hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi;
      hypre_ParAMGDDDataFACRelaxType(amgdd_data) = 3;
   }

   if (hypre_ParAMGDDDataFACRelaxType(amgdd_data) == 3)
   {
      for (level = hypre_ParAMGDDDataStartLevel(amgdd_data); level < hypre_ParAMGDataNumLevels(amg_data); level++)
      {
         hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];

         // Calculate l1_norms
         HYPRE_Int total_num_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) + hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
         hypre_AMGDDCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, total_num_nodes, hypre_AMGDDCompGridMemoryLocation(compGrid));
         hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
         hypre_CSRMatrix *offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
         for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
         {
            HYPRE_Int cf_diag = hypre_AMGDDCompGridCFMarkerArray(compGrid)[i];
            for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
            {
               if (hypre_AMGDDCompGridCFMarkerArray(compGrid)[ hypre_CSRMatrixJ(diag)[j] ] == cf_diag)
               {
                  hypre_AMGDDCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(diag)[j]);
               }
            }
            for (j = hypre_CSRMatrixI(offd)[i]; j < hypre_CSRMatrixI(offd)[i+1]; j++)
            {
               if (hypre_AMGDDCompGridCFMarkerArray(compGrid)[ hypre_CSRMatrixJ(offd)[j] + hypre_AMGDDCompGridNumOwnedNodes(compGrid) ] == cf_diag)
               {
                  hypre_AMGDDCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(offd)[j]);
               }
            }
         }
         diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
         offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
         for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedNodes(compGrid); i++)
         {
            HYPRE_Int cf_diag = hypre_AMGDDCompGridCFMarkerArray(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)];
            for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
            {
               if (hypre_AMGDDCompGridCFMarkerArray(compGrid)[ hypre_CSRMatrixJ(diag)[j] + hypre_AMGDDCompGridNumOwnedNodes(compGrid) ] == cf_diag)
               {
                  hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(diag)[j]);
               }
            }
            for (j = hypre_CSRMatrixI(offd)[i]; j < hypre_CSRMatrixI(offd)[i+1]; j++)
            {
               if (hypre_AMGDDCompGridCFMarkerArray(compGrid)[ hypre_CSRMatrixJ(offd)[j]] == cf_diag)
               {
                  hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(offd)[j]);
               }
            }
         }
      }
   }

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridFinalize( hypre_ParAMGDDData *amgdd_data )
{
   HYPRE_Int      myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   HYPRE_Int level, i, j;
   HYPRE_Int start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);

   hypre_ParAMGData *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   hypre_AMGDDCompGrid **compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);
   hypre_AMGDDCommPkg *amgddCommPkg = hypre_ParAMGDDDataCommPkg(amgdd_data);

   // Post process to remove -1 entries from matrices and reorder so that extra nodes are [real, ghost]
   for (level = start_level; level < num_levels; level++)
   {
      HYPRE_Int num_nonowned = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]);
      HYPRE_Int num_owned = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
      HYPRE_Int num_nonowned_real_nodes = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i]) num_nonowned_real_nodes++;
      }
      hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid[level]) = num_nonowned_real_nodes;
      HYPRE_Int *new_indices = hypre_CTAlloc(HYPRE_Int, num_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      HYPRE_Int real_cnt = 0;
      HYPRE_Int ghost_cnt = 0;
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_indices[i] = real_cnt++;
         }
         else new_indices[i] = num_nonowned_real_nodes + ghost_cnt++;
      }

      // Transform indices in send_flag and recv_map
      if (amgddCommPkg)
      {
         HYPRE_Int outer_level;
         for (outer_level = start_level; outer_level < num_levels; outer_level++)
         {
            HYPRE_Int proc;
            for (proc = 0; proc < hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level]; proc++)
            {
               HYPRE_Int num_send_nodes = hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
               HYPRE_Int new_num_send_nodes = 0;
               for (i = 0; i < num_send_nodes; i++)
               {
                  if (hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] >= num_owned)
                  {
                     hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][new_num_send_nodes++] = new_indices[ hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] - num_owned ] + num_owned;
                  }
                  else if (hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][new_num_send_nodes++] = hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level][i];
                  }
               }
               hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level] = new_num_send_nodes;
            }

            for (proc = 0; proc < hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level]; proc++)
            {
               HYPRE_Int num_recv_nodes = hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
               HYPRE_Int new_num_recv_nodes = 0;
               for (i = 0; i < num_recv_nodes; i++)
               {
                  if (hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][i] >= 0)
                  {
                     hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][new_num_recv_nodes++] = new_indices[hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level][i]];
                  }
               }
               hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level] = new_num_recv_nodes;
            }
         }
      }

      // Setup CF marker array and C and F masks
      hypre_AMGDDCompGridCFMarkerArray(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_owned + num_nonowned, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      if (level != num_levels-1)
      {
         // Setup CF marker array
         HYPRE_Int num_owned_c_points = 0;
         HYPRE_Int num_nonowned_real_c_points = 0;
         for (i = 0; i < num_owned; i++)
         {
            if (hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level])[i] >= 0)
            {
               hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i] = 1;
               num_owned_c_points++;
            }
            else hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i] = 0;
         }
         for (i = 0; i < num_nonowned; i++)
         {
            if (hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i] >= 0)
            {
               hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[new_indices[i] + num_owned] = 1;
               if (new_indices[i] < num_nonowned_real_nodes) num_nonowned_real_c_points++;
            }
            else hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[new_indices[i] + num_owned] = 0;
         }
         hypre_AMGDDCompGridNumOwnedCPoints(compGrid[level]) = num_owned_c_points;
         hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid[level]) = num_nonowned_real_c_points;

#if defined(HYPRE_USING_CUDA)
         // Setup owned C and F masks. NOTE: only used in the cuda version of masked matvecs.
         hypre_AMGDDCompGridOwnedCMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_owned_c_points, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridOwnedFMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_owned - num_owned_c_points, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         HYPRE_Int c_cnt = 0;
         HYPRE_Int f_cnt = 0;
         for (i = 0; i < num_owned; i++)
         {
            if (hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i])
               hypre_AMGDDCompGridOwnedCMask(compGrid[level])[c_cnt++] = i;
            else
               hypre_AMGDDCompGridOwnedFMask(compGrid[level])[f_cnt++] = i;
         }
         hypre_AMGDDCompGridNonOwnedCMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_nonowned_real_c_points, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedFMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_nonowned_real_nodes - num_nonowned_real_c_points, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         c_cnt = 0;
         f_cnt = 0;
         for (i = 0; i < num_nonowned_real_nodes; i++)
         {
            if (hypre_AMGDDCompGridCFMarkerArray(compGrid[level])[i + num_owned])
               hypre_AMGDDCompGridNonOwnedCMask(compGrid[level])[c_cnt++] = i;
            else
               hypre_AMGDDCompGridNonOwnedFMask(compGrid[level])[f_cnt++] = i;
         }
      }
      else
      {
         hypre_AMGDDCompGridNumOwnedCPoints(compGrid[level]) = 0;
         hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid[level]) = 0;
         hypre_AMGDDCompGridOwnedCMask(compGrid[level]) = NULL;
         hypre_AMGDDCompGridOwnedFMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_owned, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         for (i = 0; i < num_owned; i++) hypre_AMGDDCompGridOwnedFMask(compGrid[level])[i] = i;
         hypre_AMGDDCompGridNonOwnedCMask(compGrid[level]) = NULL;
         hypre_AMGDDCompGridNonOwnedFMask(compGrid[level]) = hypre_CTAlloc(HYPRE_Int, num_nonowned_real_nodes, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         for (i = 0; i < num_nonowned_real_nodes; i++) hypre_AMGDDCompGridNonOwnedFMask(compGrid[level])[i] = i;
#endif
      }


      // Reorder nonowned matrices
      hypre_CSRMatrix *A_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid[level]));
      hypre_CSRMatrix *A_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid[level]));

      HYPRE_Int A_diag_nnz = hypre_CSRMatrixI(A_diag)[num_nonowned];
      HYPRE_Int *new_A_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      HYPRE_Int *new_A_diag_colInd = hypre_CTAlloc(HYPRE_Int, A_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      HYPRE_Complex *new_A_diag_data = hypre_CTAlloc(HYPRE_Complex, A_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));

      HYPRE_Int A_offd_nnz = hypre_CSRMatrixI(A_offd)[num_nonowned];
      HYPRE_Int *new_A_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      HYPRE_Int *new_A_offd_colInd = hypre_CTAlloc(HYPRE_Int, A_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      HYPRE_Complex *new_A_offd_data = hypre_CTAlloc(HYPRE_Complex, A_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));

      HYPRE_Int A_real_real_nnz = 0;
      HYPRE_Int A_real_ghost_nnz = 0;

      hypre_CSRMatrix *P_diag;
      hypre_CSRMatrix *P_offd;

      HYPRE_Int P_diag_nnz;
      HYPRE_Int *new_P_diag_rowPtr;
      HYPRE_Int *new_P_diag_colInd;
      HYPRE_Complex *new_P_diag_data;

      HYPRE_Int P_offd_nnz;
      HYPRE_Int *new_P_offd_rowPtr;
      HYPRE_Int *new_P_offd_colInd;
      HYPRE_Complex *new_P_offd_data;

      hypre_CSRMatrix *R_diag;
      hypre_CSRMatrix *R_offd;

      HYPRE_Int R_diag_nnz;
      HYPRE_Int *new_R_diag_rowPtr;
      HYPRE_Int *new_R_diag_colInd;
      HYPRE_Complex *new_R_diag_data;

      HYPRE_Int R_offd_nnz;
      HYPRE_Int *new_R_offd_rowPtr;
      HYPRE_Int *new_R_offd_colInd;
      HYPRE_Complex *new_R_offd_data;

      if (level != num_levels-1 && num_nonowned)
      {
         P_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
         P_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));

         P_diag_nnz = hypre_CSRMatrixI(P_diag)[num_nonowned];
         new_P_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_P_diag_colInd = hypre_CTAlloc(HYPRE_Int, P_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_P_diag_data = hypre_CTAlloc(HYPRE_Complex, P_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));

         P_offd_nnz = hypre_CSRMatrixI(P_offd)[num_nonowned];
         new_P_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_P_offd_colInd = hypre_CTAlloc(HYPRE_Int, P_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_P_offd_data = hypre_CTAlloc(HYPRE_Complex, P_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      }
      if (hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         R_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level-1]));
         R_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level-1]));

         R_diag_nnz = hypre_CSRMatrixI(R_diag)[num_nonowned];
         new_R_diag_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_R_diag_colInd = hypre_CTAlloc(HYPRE_Int, R_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_R_diag_data = hypre_CTAlloc(HYPRE_Complex, R_diag_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));

         R_offd_nnz = hypre_CSRMatrixI(R_offd)[num_nonowned];
         new_R_offd_rowPtr = hypre_CTAlloc(HYPRE_Int, num_nonowned+1, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_R_offd_colInd = hypre_CTAlloc(HYPRE_Int, R_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         new_R_offd_data = hypre_CTAlloc(HYPRE_Complex, R_offd_nnz, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      }

      HYPRE_Int A_diag_cnt = 0;
      HYPRE_Int A_offd_cnt = 0;
      HYPRE_Int P_diag_cnt = 0;
      HYPRE_Int P_offd_cnt = 0;
      HYPRE_Int R_diag_cnt = 0;
      HYPRE_Int R_offd_cnt = 0;
      HYPRE_Int node_cnt = 0;
      // Real nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = hypre_CSRMatrixI(A_diag)[i]; j < hypre_CSRMatrixI(A_diag)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  HYPRE_Int new_col_ind = new_indices[ hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_colInd[A_diag_cnt] = new_col_ind;
                  new_A_diag_data[A_diag_cnt] = hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
                  if (new_col_ind < num_nonowned_real_nodes)
                  {
                      A_real_real_nnz++;
                  }
                  else
                  {
                      A_real_ghost_nnz++;
                  }
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = hypre_CSRMatrixI(A_offd)[i]; j < hypre_CSRMatrixI(A_offd)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels-1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = hypre_CSRMatrixI(P_diag)[i]; j < hypre_CSRMatrixI(P_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = hypre_CSRMatrixI(P_offd)[i]; j < hypre_CSRMatrixI(P_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               new_R_diag_rowPtr[node_cnt] = R_diag_cnt;
               for (j = hypre_CSRMatrixI(R_diag)[i]; j < hypre_CSRMatrixI(R_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(R_diag)[j] >= 0)
                  {
                     new_R_diag_colInd[R_diag_cnt] = hypre_CSRMatrixJ(R_diag)[j];
                     new_R_diag_data[R_diag_cnt] = hypre_CSRMatrixData(R_diag)[j];
                     R_diag_cnt++;
                  }
               }
               new_R_offd_rowPtr[node_cnt] = R_offd_cnt;
               for (j = hypre_CSRMatrixI(R_offd)[i]; j < hypre_CSRMatrixI(R_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(R_offd)[j] >= 0)
                  {
                     new_R_offd_colInd[R_offd_cnt] = hypre_CSRMatrixJ(R_offd)[j];
                     new_R_offd_data[R_offd_cnt] = hypre_CSRMatrixData(R_offd)[j];
                     R_offd_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      // Ghost nodes
      for (i = 0; i < num_nonowned; i++)
      {
         if (!hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i])
         {
            new_A_diag_rowPtr[node_cnt] = A_diag_cnt;
            for (j = hypre_CSRMatrixI(A_diag)[i]; j < hypre_CSRMatrixI(A_diag)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_diag)[j] >= 0)
               {
                  new_A_diag_colInd[A_diag_cnt] = new_indices[ hypre_CSRMatrixJ(A_diag)[j] ];
                  new_A_diag_data[A_diag_cnt] = hypre_CSRMatrixData(A_diag)[j];
                  A_diag_cnt++;
               }
            }
            new_A_offd_rowPtr[node_cnt] = A_offd_cnt;
            for (j = hypre_CSRMatrixI(A_offd)[i]; j < hypre_CSRMatrixI(A_offd)[i+1]; j++)
            {
               if (hypre_CSRMatrixJ(A_offd)[j] >= 0)
               {
                  new_A_offd_colInd[A_offd_cnt] = hypre_CSRMatrixJ(A_offd)[j];
                  new_A_offd_data[A_offd_cnt] = hypre_CSRMatrixData(A_offd)[j];
                  A_offd_cnt++;
               }
            }

            if (level != num_levels-1)
            {
               new_P_diag_rowPtr[node_cnt] = P_diag_cnt;
               for (j = hypre_CSRMatrixI(P_diag)[i]; j < hypre_CSRMatrixI(P_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_diag)[j] >= 0)
                  {
                     new_P_diag_colInd[P_diag_cnt] = hypre_CSRMatrixJ(P_diag)[j];
                     new_P_diag_data[P_diag_cnt] = hypre_CSRMatrixData(P_diag)[j];
                     P_diag_cnt++;
                  }
               }
               new_P_offd_rowPtr[node_cnt] = P_offd_cnt;
               for (j = hypre_CSRMatrixI(P_offd)[i]; j < hypre_CSRMatrixI(P_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(P_offd)[j] >= 0)
                  {
                     new_P_offd_colInd[P_offd_cnt] = hypre_CSRMatrixJ(P_offd)[j];
                     new_P_offd_data[P_offd_cnt] = hypre_CSRMatrixData(P_offd)[j];
                     P_offd_cnt++;
                  }
               }
            }
            if (hypre_ParAMGDataRestriction(amg_data) && level != 0)
            {
               new_R_diag_rowPtr[node_cnt] = R_diag_cnt;
               for (j = hypre_CSRMatrixI(R_diag)[i]; j < hypre_CSRMatrixI(R_diag)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(R_diag)[j] >= 0)
                  {
                     new_R_diag_colInd[R_diag_cnt] = hypre_CSRMatrixJ(R_diag)[j];
                     new_R_diag_data[R_diag_cnt] = hypre_CSRMatrixData(R_diag)[j];
                     R_diag_cnt++;
                  }
               }
               new_R_offd_rowPtr[node_cnt] = R_offd_cnt;
               for (j = hypre_CSRMatrixI(R_offd)[i]; j < hypre_CSRMatrixI(R_offd)[i+1]; j++)
               {
                  if (hypre_CSRMatrixJ(R_offd)[j] >= 0)
                  {
                     new_R_offd_colInd[R_offd_cnt] = hypre_CSRMatrixJ(R_offd)[j];
                     new_R_offd_data[R_offd_cnt] = hypre_CSRMatrixData(R_offd)[j];
                     R_offd_cnt++;
                  }
               }
            }
            node_cnt++;
         }
      }
      new_A_diag_rowPtr[num_nonowned] = A_diag_cnt;
      new_A_offd_rowPtr[num_nonowned] = A_offd_cnt;

      // Create these matrices, but don't initialize (will be allocated later if necessary)
      hypre_AMGDDCompGridMatrixRealReal(hypre_AMGDDCompGridA(compGrid[level])) = hypre_CSRMatrixCreate(num_nonowned_real_nodes, num_nonowned_real_nodes, A_real_real_nnz);
      hypre_AMGDDCompGridMatrixRealGhost(hypre_AMGDDCompGridA(compGrid[level])) = hypre_CSRMatrixCreate(num_nonowned_real_nodes, num_nonowned, A_real_ghost_nnz);


      if (level != num_levels-1 && num_nonowned)
      {
         new_P_diag_rowPtr[num_nonowned] = P_diag_cnt;
         new_P_offd_rowPtr[num_nonowned] = P_offd_cnt;
      }
      if (hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         new_R_diag_rowPtr[num_nonowned] = R_diag_cnt;
         new_R_offd_rowPtr[num_nonowned] = R_offd_cnt;
      }

      // Fix up P col indices on finer level
      if (level != start_level && hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level-1]))
      {
         P_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level-1]));
         P_offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level-1]));

         for (i = 0; i < hypre_CSRMatrixI(P_diag)[ hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level-1]) ]; i++)
         {
            hypre_CSRMatrixJ(P_diag)[i] = new_indices[ hypre_CSRMatrixJ(P_diag)[i] ];
         }
         // Also fix up owned offd col indices
         for (i = 0; i < hypre_CSRMatrixI(P_offd)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid[level-1]) ]; i++)
         {
            hypre_CSRMatrixJ(P_offd)[i] = new_indices[ hypre_CSRMatrixJ(P_offd)[i] ];
         }
      }
      // Fix up R col indices on this level
      if (hypre_ParAMGDataRestriction(amg_data) && level != num_levels-1 && num_nonowned)
      {
         R_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level]));
         R_offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level]));

         for (i = 0; i < hypre_CSRMatrixI(R_diag)[ hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level+1]) ]; i++)
         {
            if (hypre_CSRMatrixJ(R_diag)[i] >= 0) hypre_CSRMatrixJ(R_diag)[i] = new_indices[ hypre_CSRMatrixJ(R_diag)[i] ];
         }
         // Also fix up owned offd col indices
         for (i = 0; i < hypre_CSRMatrixI(R_offd)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid[level+1]) ]; i++)
         {
            if (hypre_CSRMatrixJ(R_offd)[i] >= 0) hypre_CSRMatrixJ(R_offd)[i] = new_indices[ hypre_CSRMatrixJ(R_offd)[i] ];
         }
      }

      // Clean up memory, deallocate old arrays and reset pointers to new arrays
      hypre_TFree(hypre_CSRMatrixI(A_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_TFree(hypre_CSRMatrixJ(A_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_TFree(hypre_CSRMatrixData(A_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_CSRMatrixI(A_diag) = new_A_diag_rowPtr;
      hypre_CSRMatrixJ(A_diag) = new_A_diag_colInd;
      hypre_CSRMatrixData(A_diag) = new_A_diag_data;
      hypre_CSRMatrixNumRows(A_diag) = num_nonowned;
      hypre_CSRMatrixNumRownnz(A_diag) = num_nonowned;
      hypre_CSRMatrixNumCols(A_diag) = num_nonowned;
      hypre_CSRMatrixNumNonzeros(A_diag) = hypre_CSRMatrixI(A_diag)[num_nonowned];

      hypre_TFree(hypre_CSRMatrixI(A_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_TFree(hypre_CSRMatrixJ(A_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_TFree(hypre_CSRMatrixData(A_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
      hypre_CSRMatrixI(A_offd) = new_A_offd_rowPtr;
      hypre_CSRMatrixJ(A_offd) = new_A_offd_colInd;
      hypre_CSRMatrixData(A_offd) = new_A_offd_data;
      hypre_CSRMatrixNumRows(A_offd) = num_nonowned;
      hypre_CSRMatrixNumRownnz(A_offd) = num_nonowned;
      hypre_CSRMatrixNumCols(A_offd) = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level]);
      hypre_CSRMatrixNumNonzeros(A_offd) = hypre_CSRMatrixI(A_offd)[num_nonowned];

      if (level != num_levels-1 && num_nonowned)
      {
         P_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));
         P_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));

         hypre_TFree(hypre_CSRMatrixI(P_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixJ(P_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixData(P_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_CSRMatrixI(P_diag) = new_P_diag_rowPtr;
         hypre_CSRMatrixJ(P_diag) = new_P_diag_colInd;
         hypre_CSRMatrixData(P_diag) = new_P_diag_data;
         hypre_CSRMatrixNumRows(P_diag) = num_nonowned;
         hypre_CSRMatrixNumRownnz(P_diag) = num_nonowned;
         hypre_CSRMatrixNumCols(P_diag) = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level+1]);
         hypre_CSRMatrixNumNonzeros(P_diag) = hypre_CSRMatrixI(P_diag)[num_nonowned];

         hypre_TFree(hypre_CSRMatrixI(P_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixJ(P_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixData(P_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_CSRMatrixI(P_offd) = new_P_offd_rowPtr;
         hypre_CSRMatrixJ(P_offd) = new_P_offd_colInd;
         hypre_CSRMatrixData(P_offd) = new_P_offd_data;
         hypre_CSRMatrixNumRows(P_offd) = num_nonowned;
         hypre_CSRMatrixNumRownnz(P_offd) = num_nonowned;
         hypre_CSRMatrixNumCols(P_offd) = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level+1]);
         hypre_CSRMatrixNumNonzeros(P_offd) = hypre_CSRMatrixI(P_offd)[num_nonowned];

         hypre_CSRMatrixNumCols(hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]))) = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level+1]);
      }
      if (hypre_ParAMGDataRestriction(amg_data) && level != 0 && num_nonowned)
      {
         R_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level-1]));
         R_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level-1]));

         hypre_TFree(hypre_CSRMatrixI(R_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixJ(R_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixData(R_diag), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_CSRMatrixI(R_diag) = new_R_diag_rowPtr;
         hypre_CSRMatrixJ(R_diag) = new_R_diag_colInd;
         hypre_CSRMatrixData(R_diag) = new_R_diag_data;
         hypre_CSRMatrixNumRows(R_diag) = num_nonowned;
         hypre_CSRMatrixNumRownnz(R_diag) = num_nonowned;
         hypre_CSRMatrixNumCols(R_diag) = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level-1]);
         hypre_CSRMatrixNumNonzeros(R_diag) = hypre_CSRMatrixI(R_diag)[num_nonowned];

         hypre_TFree(hypre_CSRMatrixI(R_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixJ(R_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_TFree(hypre_CSRMatrixData(R_offd), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_CSRMatrixI(R_offd) = new_R_offd_rowPtr;
         hypre_CSRMatrixJ(R_offd) = new_R_offd_colInd;
         hypre_CSRMatrixData(R_offd) = new_R_offd_data;
         hypre_CSRMatrixNumRows(R_offd) = num_nonowned;
         hypre_CSRMatrixNumRownnz(R_offd) = num_nonowned;
         hypre_CSRMatrixNumCols(R_offd) = hypre_AMGDDCompGridNumOwnedNodes(compGrid[level-1]);
         hypre_CSRMatrixNumNonzeros(R_offd) = hypre_CSRMatrixI(R_offd)[num_nonowned];

         hypre_CSRMatrixNumCols(hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level-1]))) = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level-1]);
      }

      // Setup comp grid vectors
      hypre_AMGDDCompGridU(compGrid[level]) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrid[level])) = hypre_ParVectorLocalVector( hypre_ParAMGDataUArray(amg_data)[level] );
      hypre_AMGDDCompGridVectorOwnsOwnedVector(hypre_AMGDDCompGridU(compGrid[level])) = 0;
      hypre_AMGDDCompGridVectorNumReal(hypre_AMGDDCompGridU(compGrid[level])) = num_nonowned_real_nodes;
      hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
      hypre_SeqVectorInitialize(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(compGrid[level])));

      hypre_AMGDDCompGridF(compGrid[level]) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridF(compGrid[level])) = hypre_ParVectorLocalVector( hypre_ParAMGDataFArray(amg_data)[level] );
      hypre_AMGDDCompGridVectorOwnsOwnedVector(hypre_AMGDDCompGridF(compGrid[level])) = 0;
      hypre_AMGDDCompGridVectorNumReal(hypre_AMGDDCompGridF(compGrid[level])) = num_nonowned_real_nodes;
      hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(compGrid[level])) = hypre_SeqVectorCreate(num_nonowned);
      hypre_SeqVectorInitialize(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridF(compGrid[level])));

      hypre_AMGDDCompGridTemp(compGrid[level]) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp(compGrid[level]), num_owned, num_nonowned, num_nonowned_real_nodes);

      if (level < num_levels)
      {
         hypre_AMGDDCompGridS(compGrid[level]) = hypre_AMGDDCompGridVectorCreate();
         hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridS(compGrid[level]), num_owned, num_nonowned, num_nonowned_real_nodes);

         hypre_AMGDDCompGridT(compGrid[level]) = hypre_AMGDDCompGridVectorCreate();
         hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridT(compGrid[level]), num_owned, num_nonowned, num_nonowned_real_nodes);
      }

      // Free up arrays we no longer need
      if (hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level]) = NULL;
      }
      if (hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid[level]) = NULL;
      }
      if (hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridOwnedCoarseIndices(compGrid[level]) = NULL;
      }
      if (hypre_AMGDDCompGridNonOwnedSort(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridNonOwnedSort(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedSort(compGrid[level]) = NULL;
      }
      if (hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]))
      {
         hypre_TFree(hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]), hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
         hypre_AMGDDCompGridNonOwnedInvSort(compGrid[level]) = NULL;
      }
      hypre_TFree(new_indices, hypre_AMGDDCompGridMemoryLocation(compGrid[level]));
   }

   // Setup R = P^T if R not specified
   if (!hypre_ParAMGDataRestriction(amg_data))
   {
      for (level = start_level; level < num_levels-1; level++)
      {
         // !!! TODO: if BoomerAMG explicitly stores R = P^T, use those matrices in
         hypre_AMGDDCompGridR(compGrid[level]) = hypre_AMGDDCompGridMatrixCreate();
         hypre_AMGDDCompGridMatrixOwnsOwnedMatrices(hypre_AMGDDCompGridR(compGrid[level])) = 1;
         hypre_CSRMatrixTranspose(hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridP(compGrid[level])),
                                  &hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridR(compGrid[level])), 1);
         if (hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]))
             hypre_CSRMatrixTranspose(hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridP(compGrid[level])),
                                  &hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level])), 1);
         hypre_CSRMatrixTranspose(hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level])),
                                  &hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridR(compGrid[level])), 1);
         if (hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]))
             hypre_CSRMatrixTranspose(hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level])),
                                  &hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level])), 1);
      }
   }

   // Finish up comm pkg
   if (amgddCommPkg)
   {
      HYPRE_Int outer_level;
      for (outer_level = start_level; outer_level < num_levels; outer_level++)
      {
         HYPRE_Int proc;
         HYPRE_Int num_send_procs = hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level];
         HYPRE_Int new_num_send_procs = 0;
         for (proc = 0; proc < num_send_procs; proc++)
         {
            hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs] = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs] += hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
            }
            if (hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[outer_level][new_num_send_procs])
            {
               hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[outer_level][new_num_send_procs] = hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[outer_level][proc];
               for (level = outer_level; level < num_levels; level++)
               {
                  hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][new_num_send_procs][level] = hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[outer_level][proc][level];
                  hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][new_num_send_procs][level] = hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[outer_level][proc][level];
               }
               new_num_send_procs++;
            }
         }
         hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[outer_level] = new_num_send_procs;

         HYPRE_Int num_recv_procs = hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level];
         HYPRE_Int new_num_recv_procs = 0;
         for (proc = 0; proc < num_recv_procs; proc++)
         {
            hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs] = 0;
            for (level = outer_level; level < num_levels; level++)
            {
               hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs] += hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
            }
            if (hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[outer_level][new_num_recv_procs])
            {
               hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[outer_level][new_num_recv_procs] = hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[outer_level][proc];
               for (level = outer_level; level < num_levels; level++)
               {
                  hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][new_num_recv_procs][level] = hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[outer_level][proc][level];
                  hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][new_num_recv_procs][level] = hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[outer_level][proc][level];
               }
               new_num_recv_procs++;
            }
         }
         hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[outer_level] = new_num_recv_procs;
      }
   }

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridResize( hypre_AMGDDCompGrid *compGrid, HYPRE_Int new_size, HYPRE_Int need_coarse_info )
{
   // This function reallocates memory to hold nonowned info for the comp grid
   HYPRE_MemoryLocation memory_location = hypre_AMGDDCompGridMemoryLocation(compGrid);
   HYPRE_Int old_size = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);

   hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedGlobalIndices(compGrid), HYPRE_Int, old_size, HYPRE_Int, new_size, memory_location);
   hypre_AMGDDCompGridNonOwnedRealMarker(compGrid) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedRealMarker(compGrid), HYPRE_Int, old_size, HYPRE_Int, new_size, memory_location);
   hypre_AMGDDCompGridNonOwnedSort(compGrid) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedSort(compGrid), HYPRE_Int, old_size, HYPRE_Int, new_size, memory_location);
   hypre_AMGDDCompGridNonOwnedInvSort(compGrid) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedInvSort(compGrid), HYPRE_Int, old_size, HYPRE_Int, new_size, memory_location);

   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
   hypre_CSRMatrixResize(nonowned_diag, new_size, new_size, hypre_CSRMatrixNumNonzeros(nonowned_diag));
   hypre_CSRMatrixResize(nonowned_offd, new_size, hypre_CSRMatrixNumCols(nonowned_offd), hypre_CSRMatrixNumNonzeros(nonowned_offd));

   if (need_coarse_info)
   {
      hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid), HYPRE_Int, old_size, HYPRE_Int, new_size, memory_location);
   }

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridSetupLocalIndices( hypre_AMGDDCompGrid **compGrid, HYPRE_Int *nodes_added_on_level, HYPRE_Int ****recv_map,
   HYPRE_Int num_recv_procs, HYPRE_Int **A_tmp_info, HYPRE_Int current_level, HYPRE_Int num_levels )
{
   // when nodes are added to a composite grid, global info is copied over, but local indices must be generated appropriately for all added nodes
   // this must be done on each level as info is added to correctly construct subsequent Psi_c grids
   // also done after each ghost layer is added
   HYPRE_Int      level,proc,i,j,k;
   HYPRE_Int      global_index, local_index, coarse_index;

   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid[current_level]);
   hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(A);
   hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
   hypre_CSRMatrix *nonowned_offd = hypre_AMGDDCompGridMatrixNonOwnedOffd(A);

   // On current_level, need to deal with A_tmp_info
   HYPRE_Int row = hypre_CSRMatrixNumCols(owned_offd)+1;
   HYPRE_Int diag_rowptr = hypre_CSRMatrixI(nonowned_diag)[ hypre_CSRMatrixNumCols(owned_offd) ];
   HYPRE_Int offd_rowptr = hypre_CSRMatrixI(nonowned_offd)[ hypre_CSRMatrixNumCols(owned_offd) ];
   for (proc = 0; proc < num_recv_procs; proc++)
   {
      HYPRE_Int cnt = 0;
      HYPRE_Int num_original_recv_dofs = A_tmp_info[proc][cnt++];
      HYPRE_Int remaining_dofs = A_tmp_info[proc][cnt++];

      for (i = 0; i < remaining_dofs; i++)
      {
         HYPRE_Int row_size = A_tmp_info[proc][cnt++];
         for (j = 0; j < row_size; j++)
         {
            HYPRE_Int incoming_index = A_tmp_info[proc][cnt++];

            // Incoming is a global index (could be owned or nonowned)
            if (incoming_index < 0)
            {
               incoming_index = -(incoming_index+1);
               // See whether global index is owned on this proc (if so, can directly setup appropriate local index)
               if (incoming_index >= hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]) && incoming_index <= hypre_AMGDDCompGridLastGlobalIndex(compGrid[current_level]))
               {
                  // Add to offd
                  if (offd_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_offd))
                     hypre_CSRMatrixResize(nonowned_offd, hypre_CSRMatrixNumRows(nonowned_offd), hypre_CSRMatrixNumCols(nonowned_offd), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_offd)));
                  hypre_CSRMatrixJ(nonowned_offd)[offd_rowptr++] = incoming_index - hypre_AMGDDCompGridFirstGlobalIndex(compGrid[current_level]);
               }
               else
               {
                  // Add to diag (global index, not in buffer, so need to do local binary search)
                  if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
                  {
                     hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)), hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
                     hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)));
                  }
                  // If we dof not found in comp grid, then mark this as a missing connection
                  hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level])[ hypre_AMGDDCompGridNumMissingColIndices(compGrid[current_level])++ ] = diag_rowptr;
                  hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = -(incoming_index+1);
               }
            }
            // Incoming is an index to dofs within the buffer (by construction, nonowned)
            else
            {
               // Add to diag (index is within buffer, so we can directly go to local index)
               if (diag_rowptr >= hypre_CSRMatrixNumNonzeros(nonowned_diag))
               {
                  hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]) = hypre_TReAlloc_v2(hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[current_level]), HYPRE_Int, hypre_CSRMatrixNumNonzeros(nonowned_diag), HYPRE_Int, ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)), hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
                  hypre_CSRMatrixResize(nonowned_diag, hypre_CSRMatrixNumRows(nonowned_diag), hypre_CSRMatrixNumCols(nonowned_diag), ceil(1.5*hypre_CSRMatrixNumNonzeros(nonowned_diag)));
               }
               local_index = recv_map[current_level][proc][current_level][ incoming_index ];
               if (local_index < 0) local_index = -(local_index + 1);
               hypre_CSRMatrixJ(nonowned_diag)[diag_rowptr++] = local_index - hypre_AMGDDCompGridNumOwnedNodes(compGrid[current_level]);
            }
         }

         // Update row pointers
         hypre_CSRMatrixI(nonowned_offd)[ row ] = offd_rowptr;
         hypre_CSRMatrixI(nonowned_diag)[ row ] = diag_rowptr;
         row++;
      }
      hypre_TFree(A_tmp_info[proc], hypre_AMGDDCompGridMemoryLocation(compGrid[current_level]));
   }
   hypre_TFree(A_tmp_info, HYPRE_MEMORY_HOST);

   // Loop over levels from current to coarsest
   for (level = current_level; level < num_levels; level++)
   {
      A = hypre_AMGDDCompGridA(compGrid[level]);
      nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);

      // If we have added nodes on this level
      if (nodes_added_on_level[level])
      {
         // Look for missing col ind connections
         HYPRE_Int num_missing_col_ind = hypre_AMGDDCompGridNumMissingColIndices(compGrid[level]);
         hypre_AMGDDCompGridNumMissingColIndices(compGrid[level]) = 0;
         for (i = 0; i < num_missing_col_ind; i++)
         {
            j = hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level])[i];
            global_index = hypre_CSRMatrixJ(nonowned_diag)[ j ];
            global_index = -(global_index+1);
            local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level], global_index);
            // If we dof not found in comp grid, then mark this as a missing connection
            if (local_index == -1)
            {
               local_index = -(global_index+1);
               hypre_AMGDDCompGridNonOwnedDiagMissingColIndices(compGrid[level])[ hypre_AMGDDCompGridNumMissingColIndices(compGrid[level])++ ] = j;
            }
            hypre_CSRMatrixJ(nonowned_diag)[ j ] = local_index;
         }
      }

      // if we are not on the coarsest level
      if (level != num_levels-1)
      {
         // loop over indices of non-owned nodes on this level
         // No guarantee that previous ghost dofs converted to real dofs have coarse local indices setup...
         // Thus we go over all non-owned dofs here instead of just the added ones, but we only setup coarse local index where necessary.
         // NOTE: can't use nodes_added_on_level here either because real overwritten by ghost doesn't count as added node (so you can miss setting these up)
         for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level]); i++)
         {
            // fix up the coarse local indices
            coarse_index = hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i];
            HYPRE_Int is_real = hypre_AMGDDCompGridNonOwnedRealMarker(compGrid[level])[i];

            // setup coarse local index if necessary
            if (coarse_index < -1 && is_real)
            {
               coarse_index = -(coarse_index+2); // Map back to regular global index
               local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level+1], coarse_index);
               hypre_AMGDDCompGridNonOwnedCoarseIndices(compGrid[level])[i] = local_index;
            }
         }
      }
   }

   return 0;
}

HYPRE_Int hypre_AMGDDCompGridSetupLocalIndicesP( hypre_ParAMGDDData *amgdd_data )
{
   HYPRE_Int myid;
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

   hypre_ParAMGData *amg_data = hypre_ParAMGDDDataAMG(amgdd_data);
   hypre_AMGDDCompGrid **compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data);
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
   HYPRE_Int start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);

   HYPRE_Int                  i,level;

   for (level = start_level; level < num_levels-1; level++)
   {
      // Setup owned offd col indices
      hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridP(compGrid[level]));

      for (i = 0; i < hypre_CSRMatrixI(owned_offd)[hypre_AMGDDCompGridNumOwnedNodes(compGrid[level])]; i++)
      {
         HYPRE_Int local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level+1], hypre_CSRMatrixJ(owned_offd)[i]);
         if (local_index == -1) hypre_CSRMatrixJ(owned_offd)[i] = -(hypre_CSRMatrixJ(owned_offd)[i] + 1);
         else hypre_CSRMatrixJ(owned_offd)[i] = local_index;
      }

      // Setup nonowned diag col indices
      hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridP(compGrid[level]));

      for (i = 0; i < hypre_CSRMatrixI(nonowned_diag)[hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level])]; i++)
      {
         HYPRE_Int local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level+1], hypre_CSRMatrixJ(nonowned_diag)[i]);
         if (local_index == -1) hypre_CSRMatrixJ(nonowned_diag)[i] = -(hypre_CSRMatrixJ(nonowned_diag)[i] + 1);
         else hypre_CSRMatrixJ(nonowned_diag)[i] = local_index;
      }
   }

   if (hypre_ParAMGDataRestriction(amg_data))
   {
       for (level = start_level; level < num_levels-1; level++)
       {
          // Setup owned offd col indices
          hypre_CSRMatrix *owned_offd = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridR(compGrid[level]));

          for (i = 0; i < hypre_CSRMatrixI(owned_offd)[hypre_AMGDDCompGridNumOwnedNodes(compGrid[level+1])]; i++)
          {
             HYPRE_Int local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level], hypre_CSRMatrixJ(owned_offd)[i]);
             if (local_index == -1) hypre_CSRMatrixJ(owned_offd)[i] = -(hypre_CSRMatrixJ(owned_offd)[i] + 1);
             else hypre_CSRMatrixJ(owned_offd)[i] = local_index;
          }

          // Setup nonowned diag col indices
          hypre_CSRMatrix *nonowned_diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridR(compGrid[level]));

          for (i = 0; i < hypre_CSRMatrixI(nonowned_diag)[hypre_AMGDDCompGridNumNonOwnedNodes(compGrid[level+1])]; i++)
          {
             HYPRE_Int local_index = hypre_AMGDDCompGridLocalIndexBinarySearch(compGrid[level], hypre_CSRMatrixJ(nonowned_diag)[i]);
             if (local_index == -1) hypre_CSRMatrixJ(nonowned_diag)[i] = -(hypre_CSRMatrixJ(nonowned_diag)[i] + 1);
             else hypre_CSRMatrixJ(nonowned_diag)[i] = local_index;
          }
       }
   }

   return 0;
}

hypre_AMGDDCommPkg* hypre_AMGDDCommPkgCreate(HYPRE_Int num_levels)
{
   hypre_AMGDDCommPkg   *amgddCommPkg;

   amgddCommPkg = hypre_CTAlloc(hypre_AMGDDCommPkg, 1, HYPRE_MEMORY_HOST);

   hypre_AMGDDCommPkgNumLevels(amgddCommPkg) = num_levels;

   hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgSendProcs(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgRecvProcs(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int*, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int**, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgSendFlag(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);
   hypre_AMGDDCommPkgRecvMap(amgddCommPkg) = hypre_CTAlloc(HYPRE_Int***, num_levels, HYPRE_MEMORY_HOST);

   return amgddCommPkg;
}

HYPRE_Int hypre_AMGDDCommPkgDestroy( hypre_AMGDDCommPkg *amgddCommPkg )
{
   HYPRE_Int         i, j, k;

   if ( hypre_AMGDDCommPkgSendProcs(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         hypre_TFree(hypre_AMGDDCommPkgSendProcs(amgddCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_AMGDDCommPkgSendProcs(amgddCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_AMGDDCommPkgRecvProcs(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         hypre_TFree(hypre_AMGDDCommPkgRecvProcs(amgddCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_AMGDDCommPkgRecvProcs(amgddCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         hypre_TFree(hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_AMGDDCommPkgSendBufferSize(amgddCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         hypre_TFree(hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_AMGDDCommPkgRecvBufferSize(amgddCommPkg), HYPRE_MEMORY_HOST);
   }

   if ( hypre_AMGDDCommPkgSendFlag(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[i]; j++)
         {
            for (k = 0; k < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
            {
               if ( hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j][k] ) hypre_TFree( hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j][k], HYPRE_MEMORY_HOST );
            }
            hypre_TFree( hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_AMGDDCommPkgSendFlag(amgddCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_AMGDDCommPkgSendFlag(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_AMGDDCommPkgRecvMap(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[i]; j++)
         {
            for (k = 0; k < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); k++)
            {
               if ( hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j][k] ) hypre_TFree( hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j][k], HYPRE_MEMORY_HOST );
            }
            hypre_TFree( hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_AMGDDCommPkgRecvMap(amgddCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_AMGDDCommPkgRecvMap(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg)[i]; j++)
         {
            hypre_TFree( hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_AMGDDCommPkgNumSendNodes(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg) )
   {
      for (i = 0; i < hypre_AMGDDCommPkgNumLevels(amgddCommPkg); i++)
      {
         for (j = 0; j < hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg)[i]; j++)
         {
            hypre_TFree( hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[i][j], HYPRE_MEMORY_HOST );
         }
         hypre_TFree( hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg)[i], HYPRE_MEMORY_HOST );
      }
      hypre_TFree( hypre_AMGDDCommPkgNumRecvNodes(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg) )
   {
      hypre_TFree( hypre_AMGDDCommPkgNumSendProcs(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   if ( hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg) )
   {
      hypre_TFree( hypre_AMGDDCommPkgNumRecvProcs(amgddCommPkg), HYPRE_MEMORY_HOST );
   }

   hypre_TFree(amgddCommPkg, HYPRE_MEMORY_HOST);

   return 0;
}

