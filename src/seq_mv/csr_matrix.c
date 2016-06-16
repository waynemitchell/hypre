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

/******************************************************************************
 *
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#ifdef HYPRE_PROFILE
HYPRE_Real hypre_profile_times[HYPRE_TIMER_ID_COUNT] = { 0 };
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixCreate( HYPRE_Int num_rows,
                       HYPRE_Int num_cols,
                       HYPRE_Int num_nonzeros )
{
   hypre_CSRMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_CSRMatrix, 1);

   hypre_CSRMatrixData(matrix) = NULL;
   hypre_CSRMatrixI(matrix)    = NULL;
   hypre_CSRMatrixJ(matrix)    = NULL;
   hypre_CSRMatrixRownnz(matrix) = NULL;
   hypre_CSRMatrixNumRows(matrix) = num_rows;
   hypre_CSRMatrixNumCols(matrix) = num_cols;
   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;

   /* set defaults */
   hypre_CSRMatrixOwnsData(matrix) = 1;
   hypre_CSRMatrixNumRownnz(matrix) = num_rows;
#ifdef HYPRE_USE_CUDA
   hypre_CSRMatrixDevice(matrix) = NULL;
#endif

   return matrix;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_CSRMatrixDestroy( hypre_CSRMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   if (matrix)
   {
     
#ifdef HYPRE_USE_CUDA
     cuda_MatrixDestroy(matrix);
#endif
      hypre_TFree(hypre_CSRMatrixI(matrix));
      hypre_CSRMatrixI(matrix)    = NULL;
      if (hypre_CSRMatrixRownnz(matrix))
         hypre_TFree(hypre_CSRMatrixRownnz(matrix));
      if ( hypre_CSRMatrixOwnsData(matrix) )
      {
         hypre_TFree(hypre_CSRMatrixData(matrix));
         hypre_TFree(hypre_CSRMatrixJ(matrix));
         hypre_CSRMatrixData(matrix) = NULL;
         hypre_CSRMatrixJ(matrix)    = NULL;
      }
      hypre_TFree(matrix);
      matrix = NULL;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_CSRMatrixInitialize( hypre_CSRMatrix *matrix )
{
   HYPRE_Int  num_rows     = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int  num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
/*   HYPRE_Int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix); */

   HYPRE_Int  ierr=0;

   if ( ! hypre_CSRMatrixData(matrix) && num_nonzeros )
      hypre_CSRMatrixData(matrix) = hypre_CTAlloc(HYPRE_Complex, num_nonzeros);
   if ( ! hypre_CSRMatrixI(matrix) )
      hypre_CSRMatrixI(matrix)    = hypre_CTAlloc(HYPRE_Int, num_rows + 1);
/*   if ( ! hypre_CSRMatrixRownnz(matrix) )
     hypre_CSRMatrixRownnz(matrix)    = hypre_CTAlloc(HYPRE_Int, num_rownnz);*/
   if ( ! hypre_CSRMatrixJ(matrix) && num_nonzeros )
      hypre_CSRMatrixJ(matrix)    = hypre_CTAlloc(HYPRE_Int, num_nonzeros);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_CSRMatrixSetDataOwner( hypre_CSRMatrix *matrix,
                             HYPRE_Int              owns_data )
{
   HYPRE_Int    ierr=0;

   hypre_CSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSetRownnz( hypre_CSRMatrix *matrix )
{
   HYPRE_Int    ierr=0;
   HYPRE_Int  num_rows = hypre_CSRMatrixNumRows(matrix);
   HYPRE_Int  *A_i = hypre_CSRMatrixI(matrix);
   HYPRE_Int  *Arownnz;

   HYPRE_Int i, adiag;
   HYPRE_Int irownnz=0;

   for (i=0; i < num_rows; i++)
   {
      adiag = (A_i[i+1] - A_i[i]);
      if(adiag > 0) irownnz++;
   }

   hypre_CSRMatrixNumRownnz(matrix) = irownnz;

   if ((irownnz == 0) || (irownnz == num_rows))
   {
      hypre_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = hypre_CTAlloc(HYPRE_Int, irownnz);
      irownnz = 0;
      for (i=0; i < num_rows; i++)
      {
         adiag = A_i[i+1]-A_i[i];
         if(adiag > 0) Arownnz[irownnz++] = i;
      }
      hypre_CSRMatrixRownnz(matrix) = Arownnz;
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixRead( char *file_name )
{
   hypre_CSRMatrix  *matrix;

   FILE    *fp;

   HYPRE_Complex *matrix_data;
   HYPRE_Int     *matrix_i;
   HYPRE_Int     *matrix_j;
   HYPRE_Int      num_rows;
   HYPRE_Int      num_nonzeros;
   HYPRE_Int      max_col = 0;

   HYPRE_Int      file_base = 1;
   
   HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &num_rows);

   matrix_i = hypre_CTAlloc(HYPRE_Int, num_rows + 1);
   for (j = 0; j < num_rows+1; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize(matrix);

   matrix_j = hypre_CSRMatrixJ(matrix);
   for (j = 0; j < num_nonzeros; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = hypre_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   hypre_CSRMatrixNumCols(matrix) = ++max_col;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrint( hypre_CSRMatrix *matrix,
                      char            *file_name )
{
   FILE    *fp;

   HYPRE_Complex *matrix_data;
   HYPRE_Int     *matrix_i;
   HYPRE_Int     *matrix_j;
   HYPRE_Int      num_rows;
   
   HYPRE_Int      file_base = 1;
   
   HYPRE_Int      j;

   HYPRE_Int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%.14e , %.14e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrintHB: print a CSRMatrix in Harwell-Boeing format
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPrintHB( hypre_CSRMatrix *matrix_input,
                        char            *file_name )
{
   FILE            *fp;
   hypre_CSRMatrix *matrix;
   HYPRE_Complex   *matrix_data;
   HYPRE_Int       *matrix_i;
   HYPRE_Int       *matrix_j;
   HYPRE_Int        num_rows;
   HYPRE_Int        file_base = 1;
   HYPRE_Int        j, totcrd, ptrcrd, indcrd, valcrd, rhscrd;
   HYPRE_Int        ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   /* First transpose the input matrix, since HB is in CSC format */
   hypre_CSRMatrixTranspose(matrix_input, &matrix, 1);

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%-70s  Key     \n", "Title");
   ptrcrd = num_rows;
   indcrd = matrix_i[num_rows];
   valcrd = matrix_i[num_rows];
   rhscrd = 0;
   totcrd = ptrcrd + indcrd + valcrd + rhscrd;
   hypre_fprintf (fp, "%14d%14d%14d%14d%14d\n",
                  totcrd, ptrcrd, indcrd, valcrd, rhscrd);
   hypre_fprintf (fp, "%-14s%14i%14i%14i%14i\n", "RUA",
                  num_rows, num_rows, valcrd, 0);
   hypre_fprintf (fp, "%-16s%-16s%-16s%26s\n", "(1I8)", "(1I8)", "(1E16.8)", "");

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef HYPRE_COMPLEX
         hypre_fprintf(fp, "%16.8e , %16.8e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%16.8e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   hypre_CSRMatrixDestroy(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCopy:
 * copys A to B, 
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions of A and B match !!! 
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_CSRMatrixCopy( hypre_CSRMatrix *A, hypre_CSRMatrix *B, HYPRE_Int copy_data )
{
   HYPRE_Int      ierr=0;
   HYPRE_Int      num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int     *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_data;
   HYPRE_Int     *B_i = hypre_CSRMatrixI(B);
   HYPRE_Int     *B_j = hypre_CSRMatrixJ(B);
   HYPRE_Complex *B_data;
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);

   HYPRE_Int i, j;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i=0; i <= num_rows; i++)
   {
      B_i[i] = A_i[i];
   }
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (j = 0; j < num_nonzeros; ++j)
   {
      B_j[j] = A_j[j];
   }

   if (copy_data)
   {
      A_data = hypre_CSRMatrixData(A);
      B_data = hypre_CSRMatrixData(B);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (j=0; j < num_nonzeros; j++)
      {
         B_data[j] = A_data[j];
      }
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone
 * Creates and returns a new copy of the argument, A.
 * Data is not copied, only structural information is reproduced.
 * Copying is a deep copy in that no pointers are copied; new arrays are
 * created where necessary.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * hypre_CSRMatrixClone( hypre_CSRMatrix * A )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows( A );
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols( A );
   HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros( A );
   hypre_CSRMatrix * B = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   HYPRE_Int * A_i;
   HYPRE_Int * A_j;
   HYPRE_Int * B_i;
   HYPRE_Int * B_j;
   HYPRE_Int i, j;

   hypre_CSRMatrixInitialize( B );

   A_i = hypre_CSRMatrixI(A);
   A_j = hypre_CSRMatrixJ(A);
   B_i = hypre_CSRMatrixI(B);
   B_j = hypre_CSRMatrixJ(B);

   for ( i=0; i<num_rows+1; ++i )  B_i[i] = A_i[i];
   for ( j=0; j<num_nonzeros; ++j )  B_j[j] = A_j[j];
   hypre_CSRMatrixNumRownnz(B) =  hypre_CSRMatrixNumRownnz(A);
   if ( hypre_CSRMatrixRownnz(A) ) hypre_CSRMatrixSetRownnz( B );

   return B;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixUnion
 * Creates and returns a matrix whose elements are the union of those of A and B.
 * Data is not computed, only structural information is created.
 * A and B must have the same numbers of rows.
 * Nothing is done about Rownnz.
 *
 * If col_map_offd_A and col_map_offd_B are zero, A and B are expected to have
 * the same column indexing.  Otherwise, col_map_offd_A, col_map_offd_B should
 * be the arrays of that name from two ParCSRMatrices of which A and B are the
 * offd blocks.
 *
 * The algorithm can be expected to have reasonable efficiency only for very
 * sparse matrices (many rows, few nonzeros per row).
 * The nonzeros of a computed row are NOT necessarily in any particular order.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * hypre_CSRMatrixUnion(
   hypre_CSRMatrix * A, hypre_CSRMatrix * B,
   HYPRE_Int * col_map_offd_A, HYPRE_Int * col_map_offd_B, HYPRE_Int ** col_map_offd_C )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows( A );
   HYPRE_Int num_cols_A = hypre_CSRMatrixNumCols( A );
   HYPRE_Int num_cols_B = hypre_CSRMatrixNumCols( B );
   HYPRE_Int num_cols;
   HYPRE_Int num_nonzeros;
   HYPRE_Int * A_i = hypre_CSRMatrixI(A);
   HYPRE_Int * A_j = hypre_CSRMatrixJ(A);
   HYPRE_Int * B_i = hypre_CSRMatrixI(B);
   HYPRE_Int * B_j = hypre_CSRMatrixJ(B);
   HYPRE_Int * C_i;
   HYPRE_Int * C_j;
   HYPRE_Int * jC = NULL;
   HYPRE_Int i, jA, jB, jBg;
   HYPRE_Int ma, mb, mc, ma_min, ma_max, match;
   hypre_CSRMatrix * C;

   hypre_assert( num_rows == hypre_CSRMatrixNumRows(B) );
   if ( col_map_offd_B ) hypre_assert( col_map_offd_A );
   if ( col_map_offd_A ) hypre_assert( col_map_offd_B );

   /* ==== First, go through the columns of A and B to count the columns of C. */
   if ( col_map_offd_A==0 )
   {  /* The matrices are diagonal blocks.
         Normally num_cols_A==num_cols_B, col_starts is the same, etc.
      */
      num_cols = hypre_max( num_cols_A, num_cols_B );
   }
   else
   {  /* The matrices are offdiagonal blocks. */
      jC = hypre_CTAlloc( HYPRE_Int, num_cols_B );
      num_cols = num_cols_A;  /* initialization; we'll compute the actual value */
      for ( jB=0; jB<num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma=0; ma<num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma]==jBg )
               match = 1;
         }
         if ( match==0 )
         {
            jC[jB] = num_cols;
            ++num_cols;
         }
      }
   }

   /* ==== If we're working on a ParCSRMatrix's offd block,
      make and load col_map_offd_C */
   if ( col_map_offd_A )
   {
      *col_map_offd_C = hypre_CTAlloc( HYPRE_Int, num_cols );
      for ( jA=0; jA<num_cols_A; ++jA )
         (*col_map_offd_C)[jA] = col_map_offd_A[jA];
      for ( jB=0; jB<num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma=0; ma<num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma]==jBg )
               match = 1;
         }
         if ( match==0 )
            (*col_map_offd_C)[ jC[jB] ] = jBg;
      }
   }


   /* ==== The first run through A and B is to count the number of nonzero elements,
      without HYPRE_Complex-counting duplicates.  Then we can create C. */
   num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   for ( i=0; i<num_rows; ++i )
   {
      ma_min = A_i[i];  ma_max = A_i[i+1];
      for ( mb=B_i[i]; mb<B_i[i+1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B ) jB = col_map_offd_B[jB];
         match = 0;
         for ( ma=ma_min; ma<ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A ) jA = col_map_offd_A[jA];
            if ( jB == jA )
            {
               match = 1;
               if( ma==ma_min ) ++ma_min;
               break;
            }
         }
         if ( match==0 )
            ++num_nonzeros;
      }
   }

   C = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   hypre_CSRMatrixInitialize( C );


   /* ==== The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
   C_i = hypre_CSRMatrixI(C);
   C_i[0] = 0;
   C_j = hypre_CSRMatrixJ(C);
   mc = 0;
   for ( i=0; i<num_rows; ++i )
   {
      ma_min = A_i[i];  ma_max = A_i[i+1];
      for ( ma=ma_min; ma<ma_max; ++ma )
      {
         C_j[mc] = A_j[ma];
         ++mc;
      }
      for ( mb=B_i[i]; mb<B_i[i+1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B ) jB = col_map_offd_B[jB];
         match = 0;
         for ( ma=ma_min; ma<ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A ) jA = col_map_offd_A[jA];
            if ( jB == jA )
            {
               match = 1;
               if( ma==ma_min ) ++ma_min;
               break;
            }
         }
         if ( match==0 )
         {
            if ( col_map_offd_A )
               C_j[mc] = jC[ B_j[mb] ];
            else
               C_j[mc] = B_j[mb];
            /* ... I don't know whether column indices are required to be in any
               particular order.  If so, we'll need to sort. */
            ++mc;
         }
      }
      C_i[i+1] = mc;
   }

   hypre_assert( mc == num_nonzeros );
   if (jC) hypre_TFree( jC );

   return C;
}

static HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBoundary(hypre_CSRMatrix *A, HYPRE_Int idx)
{
   HYPRE_Int num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int num_rowsA = hypre_CSRMatrixNumRows(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);

   HYPRE_Int num_threads = hypre_NumActiveThreads();

   HYPRE_Int nonzeros_per_thread = (num_nonzerosA + num_threads - 1)/num_threads;

   if (idx <= 0)
   {
      return 0;
   }
   else if (idx >= num_threads)
   {
      return num_rowsA;
   }
   else
   {
      return (HYPRE_Int)(hypre_LowerBound(A_i, A_i + num_rowsA, nonzeros_per_thread*idx) - A_i);
   }
}

HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum());
}

HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum() + 1);
}

#ifdef HYPRE_USE_CUDA
  
void hypre_CSRMatrixMapToDevice(hypre_CSRMatrix *A){
  
  if (hypre_CSRMatrixDevice(A)!=NULL){
    printf("ERROR:: Trying to map an already mapped file\n");
  }

  
  // Allocate memory for the struct for holding all the device pointers
  if (hypre_CSRMatrixDevice(A)==NULL) {
    hypre_CSRMatrixDevice(A)=hypre_CTAlloc(cuda_CSRMatrix, 1);
    hypre_CSRMatrixDataDevice(A)=NULL;
    hypre_CSRMatrixIDevice(A)=NULL;
    hypre_CSRMatrixJDevice(A)=NULL;
    hypre_CSRMatrixCopiedToDevice(A)=0;
  }

  // Get a cuSPARSE handle 
  cusparseStatus_t status;
  status= cusparseCreate(&(hypre_CSRMatrixHandle(A)));
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR:: CUSPARSE Library initialization failed\n");
    hypre_CSRMatrixHandle(A)=0;
  } 
  // Create and Set Matrix Desciptor
  status= cusparseCreateMatDescr(&(hypre_CSRMatrixDescr(A))); 
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR:: Matrix descriptor initialization failed\n");
    //cudaDeviceReset();
    //return 1;
  } 
  cusparseSetMatType(hypre_CSRMatrixDescr(A),CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(hypre_CSRMatrixDescr(A),CUSPARSE_INDEX_BASE_ZERO);

  
  // Allocate device memory for Data and row and column vectors
  HYPRE_Int  num_rows     = hypre_CSRMatrixNumRows(A);
  HYPRE_Int  num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
  
  if ( ! hypre_CSRMatrixDataDevice(A) && num_nonzeros ){
    gpuErrchk(cudaMalloc((void**)&(hypre_CSRMatrixDataDevice(A)),num_nonzeros*sizeof(HYPRE_Complex)));
  } 
  

  if ( ! hypre_CSRMatrixIDevice(A) ){
    gpuErrchk(cudaMalloc((void**)&hypre_CSRMatrixIDevice(A),(num_rows+1)*sizeof(HYPRE_Int)));
    
  }

  if ( ! hypre_CSRMatrixJDevice(A) && num_nonzeros ){
    gpuErrchk(cudaMalloc((void**)&hypre_CSRMatrixJDevice(A),(num_nonzeros)*sizeof(HYPRE_Int)));
  }

}

// Some copy routines. Probably needs to be in a separate standalone file


void hypre_CSRMatrixH2D(hypre_CSRMatrix *matrix){
 
  hypre_CSRMatrixDataH2D(matrix);
  hypre_CSRMatrixIH2D(matrix);
  hypre_CSRMatrixJH2D(matrix);
  
 }
  
void hypre_CSRMatrixDataH2D(hypre_CSRMatrix *matrix){
  PUSH_RANGE("MatDataSend",0);
  gpuErrchk(cudaMemcpy(hypre_CSRMatrixDataDevice(matrix),hypre_CSRMatrixData(matrix), 
			(size_t)(matrix->num_nonzeros*sizeof(HYPRE_Complex)), 
			cudaMemcpyHostToDevice));
  POP_RANGE;
}

void hypre_CSRMatrixIH2D(hypre_CSRMatrix *matrix){
  PUSH_RANGE("MatISend",1);
  gpuErrchk(cudaMemcpy(hypre_CSRMatrixIDevice(matrix), hypre_CSRMatrixI(matrix),
		       (size_t)((matrix->num_rows+1)*sizeof(HYPRE_Int)), 
		       cudaMemcpyHostToDevice));
  POP_RANGE;
}

void hypre_CSRMatrixJH2D(hypre_CSRMatrix *matrix){
  PUSH_RANGE("MatJSend",2);
  gpuErrchk(cudaMemcpy(hypre_CSRMatrixJDevice(matrix),hypre_CSRMatrixJ(matrix),
		       (size_t)(matrix->num_nonzeros*sizeof(HYPRE_Int)), 
		       cudaMemcpyHostToDevice));
  POP_RANGE;
}

// The Asynchrononous versions of the ones above

void hypre_CSRMatrixH2DAsync(hypre_CSRMatrix *matrix,cudaStream_t s){
 
  hypre_CSRMatrixDataH2DAsync(matrix,s);
  hypre_CSRMatrixIH2DAsync(matrix,s);
  hypre_CSRMatrixJH2DAsync(matrix,s);
  
 }
  
void hypre_CSRMatrixDataH2DAsync(hypre_CSRMatrix *matrix,cudaStream_t s){
  PUSH_RANGE("MatDataSendAsync",0);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixDataDevice(matrix),hypre_CSRMatrixData(matrix), 
			(size_t)(matrix->num_nonzeros*sizeof(HYPRE_Complex)), 
			    cudaMemcpyHostToDevice,s));
  POP_RANGE;
}

void hypre_CSRMatrixIH2DAsync(hypre_CSRMatrix *matrix,cudaStream_t s){
  PUSH_RANGE("MatISendAsync",1);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixIDevice(matrix), hypre_CSRMatrixI(matrix),
		       (size_t)((matrix->num_rows+1)*sizeof(HYPRE_Int)), 
			    cudaMemcpyHostToDevice,s));
  POP_RANGE;
}

void hypre_CSRMatrixJH2DAsync(hypre_CSRMatrix *matrix,cudaStream_t s){
  PUSH_RANGE("MatJSendAsync",2);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixJDevice(matrix),hypre_CSRMatrixJ(matrix),
		       (size_t)(matrix->num_nonzeros*sizeof(HYPRE_Int)), 
		       cudaMemcpyHostToDevice,s));
  POP_RANGE;
}
// The Asynchrononous Partial versions 

void hypre_CSRMatrixH2DAsyncPartial(hypre_CSRMatrix *matrix,float frac, cudaStream_t s){
 
  size_t rows=(size_t)(matrix->num_rows*frac);
  size_t nnz=matrix->i[(int)(matrix->num_rows*frac)];
  //nnz=hypre_CSRMatrixNumNonzeros(matrix);
  HYPRE_Int nnz_send=nnz;
  //printf("H2DAysncPartial rows %d (%d) NNZ %d (%d)\n",matrix->num_rows,rows,matrix->i[rows],nnz);
  hypre_CSRMatrixDataH2DAsyncPartial(matrix,nnz,s);
  hypre_CSRMatrixIH2DAsyncPartial(matrix,rows,s);
  /* Update nrows+1 with the correct NNZ value */
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixIDevice(matrix)+rows, &nnz_send,
			    (size_t)(sizeof(HYPRE_Int)), 
			    cudaMemcpyHostToDevice,s));
  hypre_CSRMatrixJH2DAsyncPartial(matrix,nnz, s);
  
 }
  
void hypre_CSRMatrixDataH2DAsyncPartial(hypre_CSRMatrix *matrix,size_t size,cudaStream_t s){
  PUSH_RANGE("MatDataSendAsync",0);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixDataDevice(matrix),hypre_CSRMatrixData(matrix), 
			(size_t)(size*sizeof(HYPRE_Complex)), 
			    cudaMemcpyHostToDevice,s));
  POP_RANGE;
}

void hypre_CSRMatrixIH2DAsyncPartial(hypre_CSRMatrix *matrix,size_t size, cudaStream_t s){
  PUSH_RANGE("MatISendAsync",1);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixIDevice(matrix), hypre_CSRMatrixI(matrix),
		       (size_t)(size*sizeof(HYPRE_Int)), 
			    cudaMemcpyHostToDevice,s));
  POP_RANGE;
}

void hypre_CSRMatrixJH2DAsyncPartial(hypre_CSRMatrix *matrix,size_t size, cudaStream_t s){
  PUSH_RANGE("MatJSendAsync",2);
  gpuErrchk(cudaMemcpyAsync(hypre_CSRMatrixJDevice(matrix),hypre_CSRMatrixJ(matrix),
		       (size_t)(size*sizeof(HYPRE_Int)), 
		       cudaMemcpyHostToDevice,s));
  POP_RANGE;
}


// dtor for the data allocated o the device
void cuda_MatrixDestroy(hypre_CSRMatrix *matrix){
  if (hypre_CSRMatrixDevice(matrix)){
    //printf("Destorying CSR matrix on ccuda %d \n",hypre_CSRMatrixNumNonzeros(matrix));
    gpuErrchk(cudaFree(hypre_CSRMatrixDataDevice(matrix)));
    gpuErrchk(cudaFree(hypre_CSRMatrixIDevice(matrix)));
    gpuErrchk(cudaFree(hypre_CSRMatrixJDevice(matrix)));
    if (hypre_CSRMatrixDevice(matrix)->l1_norms_device) gpuErrchk(cudaFree(hypre_CSRMatrixDevice(matrix)->l1_norms_device));
    gpuErrchk(cusparseDestroyMatDescr(hypre_CSRMatrixDescr(matrix)));
    gpuErrchk(cusparseDestroy(hypre_CSRMatrixHandle(matrix)));
    hypre_TFree(hypre_CSRMatrixDevice(matrix));
    hypre_CSRMatrixDevice(matrix)=NULL;
  }
}
#endif
