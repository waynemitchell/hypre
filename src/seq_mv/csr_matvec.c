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
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include <assert.h>
#include <stdio.h>
#define gpuErrchk4(ans) { gpuAssert4((ans), __FILE__, __LINE__); }
inline void gpuAssert4(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlace( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{

#ifdef HYPRE_USE_CUDA


#define CUDA_MATVEC_CUTOFF 5000000						
  if (hypre_CSRMatrixNumNonzeros(A)>CUDA_MATVEC_CUTOFF)
    return hypre_CSRMatrixMatvecOutOfPlaceHybrid2(alpha,A,x,beta,b,y,offset);
  //printf("Matrix Vector OOP %d   %d \n",hypre_CSRMatrixNumNonzeros(A),hypre_VectorSize(y));
  //if (hypre_VectorSize(y)>CUDA_MATVEC_CUTOFF)
  // return hypre_CSRMatrixMatvecOutOfPlaceHybrid2(alpha,A,x,beta,b,y,offset);
  //printf("MATVEC_OOP on Host vec size %d\n",hypre_VectorSize(y));
#endif


#ifdef HYPRE_PROFILE
   HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   //if (y!=b) printf("Ye Olde Matvec call %d\n",offset);
   //else printf("OFFSET Matvec call %d\n",offset);
   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows*num_vectors; i++)
         y_data[i] *= beta;

#ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
   
   temp = beta / alpha;
   
/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows) || num_vectors > 1)
   {
      /*-----------------------------------------------------------------------
       * y = (beta/alpha)*y
       *-----------------------------------------------------------------------*/
     
      if (temp != 1.0)
      {
         if (temp == 0.0)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = 0.0;
         }
         else
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows*num_vectors; i++)
               y_data[i] = b_data[i]*temp;
         }
      }


      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      if (num_rownnz < xpar*(num_rows))
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,m,tempx) HYPRE_SMP_SCHEDULE
#endif

         for (i = 0; i < num_rownnz; i++)
         {
            m = A_rownnz[i];

            /*
             * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
             * {
             *         j = A_j[jj];
             *  y_data[m] += A_data[jj] * x_data[j];
             * } */
            if ( num_vectors==1 )
            {
               tempx = 0;
               for (jj = A_i[m]; jj < A_i[m+1]; jj++)
                  tempx +=  A_data[jj] * x_data[A_j[jj]];
               y_data[m] += tempx;
            }
            else
               for ( j=0; j<num_vectors; ++j )
               {
                  tempx = 0;
                  for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
                     tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
                  y_data[ j*vecstride_y + m*idxstride_y] += tempx;
               }
         }
      }
      else // num_vectors > 1
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_vectors; ++j)
            {
               tempx = 0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
               }
               y_data[ j*vecstride_y + i*idxstride_y ] += tempx;
            }
         }
      }

      /*-----------------------------------------------------------------
       * y = alpha*y
       *-----------------------------------------------------------------*/

      if (alpha != 1.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows*num_vectors; i++)
            y_data[i] *= alpha;
      }
   }
   else
   { // JSP: this is currently the only path optimized
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,tempx)
#endif
      {
      HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
      HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
      hypre_assert(iBegin <= iEnd);
      hypre_assert(iBegin >= 0 && iBegin <= num_rows);
      hypre_assert(iEnd >= 0 && iEnd <= num_rows);

      if (0 == temp)
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = 0.0;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*A*x
      } // temp == 0
      else if (-1 == temp) // beta == -alpha
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x - y
         else if (-1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x + y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x - y)
      } // temp == -1
      else if (1 == temp)
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x + y
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x - y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x + y)
      }
      else
      {
         if (1 == alpha) // JSP: a common path
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = A*x + temp*y
         else if (-1 == alpha)
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = -b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx -= A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = tempx;
            }
         } // y = -A*x - temp*y
         else
         {
            for (i = iBegin; i < iEnd; i++)
            {
               tempx = b_data[i]*temp;
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  tempx += A_data[jj] * x_data[A_j[jj]];
               }
               y_data[i] = alpha*tempx;
            }
         } // y = alpha*(A*x + temp*y)
      } // temp != 0 && temp != -1 && temp != 1
      } // omp parallel
   }

   if (x == y) hypre_SeqVectorDestroy(x_tmp);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif
   return ierr;
}

HYPRE_Int
hypre_CSRMatrixMatvec( HYPRE_Complex    alpha,
                       hypre_CSRMatrix *A,
                       hypre_Vector    *x,
                       HYPRE_Complex    beta,
                       hypre_Vector    *y     )
{
#ifndef HYPRE_USE_CUDA
  return hypre_CSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y, 0);
#else
  //printf("Matrix Vector %d   %d \n",hypre_CSRMatrixNumNonzeros(A),hypre_VectorSize(y));
  if (hypre_CSRMatrixNumNonzeros(A)>CUDA_MATVEC_CUTOFF){
  //if (hypre_VectorSize(y)>CUDA_MATVEC_CUTOFF){
    //printf("MATVEC on device %d\n",hypre_VectorSize(y));
    return hypre_CSRMatrixMatvecDevice(alpha,A,x,beta,y);
  }
  else{
    //printf("MATVEC on host %d\n",hypre_VectorSize(y));
    return hypre_CSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y, 0);
  }
#endif
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvecT( HYPRE_Complex    alpha,
                        hypre_CSRMatrix *A,
                        hypre_Vector    *x,
                        HYPRE_Complex    beta,
                        hypre_Vector    *y     )
{
   HYPRE_Complex    *A_data    = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp;

   HYPRE_Complex    *y_data_expand;
   HYPRE_Int         my_thread_num = 0, offset = 0;
   
   HYPRE_Int         i, j, jv, jj;
   HYPRE_Int         num_threads;

   HYPRE_Int         ierr  = 0;

   hypre_Vector     *x_tmp = NULL;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
 
   if (num_rows != x_size)
      ierr = 1;

   if (num_cols != y_size)
      ierr = 2;

   if (num_rows != x_size && num_cols != y_size)
      ierr = 3;
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
         y_data[i] *= beta;

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;
   
   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols*num_vectors; i++)
            y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols*num_vectors; i++)
            y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = hypre_NumThreads();
   if (num_threads > 1)
   {
      y_data_expand = hypre_CTAlloc(HYPRE_Complex, num_threads*y_size);

      if ( num_vectors==1 )
      {

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,j,my_thread_num,offset)
#endif
         {                                      
            my_thread_num = hypre_GetThreadNum();
            offset =  y_size*my_thread_num;
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data_expand[offset + j] += A_data[jj] * x_data[i];
               }
            }

            /* implied barrier (for threads)*/           
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < y_size; i++)
            {
               for (j = 0; j < num_threads; j++)
               {
                  y_data[i] += y_data_expand[j*y_size + i];
                  
               }
            }

         } /* end parallel threaded region */
      }
      else
      {
         /* multiple vector case is not threaded */
         for (i = 0; i < num_rows; i++)
         {
            for ( jv=0; jv<num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
               }
            }
         }
      }

      hypre_TFree(y_data_expand);

   }
   else 
   {
      for (i = 0; i < num_rows; i++)
      {
         if ( num_vectors==1 )
         {
            for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            {
               j = A_j[jj];
               y_data[j] += A_data[jj] * x_data[i];
            }
         }
         else
         {
            for ( jv=0; jv<num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j*idxstride_y + jv*vecstride_y ] +=
                     A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
               }
            }
         }
      }
   }
   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols*num_vectors; i++)
         y_data[i] *= alpha;
   }

   if (x == y) hypre_SeqVectorDestroy(x_tmp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMatvec_FF( HYPRE_Complex    alpha,
                          hypre_CSRMatrix *A,
                          hypre_Vector    *x,
                          HYPRE_Complex    beta,
                          hypre_Vector    *y,
                          HYPRE_Int       *CF_marker_x,
                          HYPRE_Int       *CF_marker_y,
                          HYPRE_Int        fpt )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         y_size = hypre_VectorSize(y);

   HYPRE_Complex      temp;

   HYPRE_Int         i, jj;

   HYPRE_Int         ierr = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size)
      ierr = 2;

   if (num_cols != x_size && num_rows != y_size)
      ierr = 3;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) y_data[i] *= beta;

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) y_data[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj) HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      if (CF_marker_x[i] == fpt)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i+1]; jj++)
            if (CF_marker_y[A_j[jj]] == fpt) temp += A_data[jj] * x_data[A_j[jj]];
         y_data[i] = temp;
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
   }

   return ierr;
}



#ifdef HYPRE_USE_CUDA
HYPRE_Int
hypre_CSRMatrixMatvecDevice( HYPRE_Complex    alpha,
                       hypre_CSRMatrix *A,
                       hypre_Vector    *x,
                       HYPRE_Complex    beta,
                       hypre_Vector    *y     ){
  PUSH_RANGE("Matvec DEVICE",4);
  //printf("Entre hypre_CSRMatrixMatvec CUDA Version %d %d %d\n",A->num_rows,A->num_nonzeros,A->i[A->num_rows]);
  //printf("Size of data varbls is %d Alpha = %lf, beta = %lf \n",sizeof(HYPRE_Complex),alpha,beta);
  if (!(hypre_CSRMatrixDevice(A)))hypre_CSRMatrixMapToDevice(A);
  if (!hypre_VectorDevice(x)) hypre_VectorMapToDevice(x);
  if (!hypre_VectorDevice(y)) hypre_VectorMapToDevice(y);
  // printf("IN CUDAFIED hypre_CSRMatrixMatvec\n");
  y->ref_count++;
  if (!hypre_CSRMatrixCopiedToDevice(A)){
    hypre_CSRMatrixH2D(A);
    hypre_CSRMatrixCopiedToDevice(A)=1;
#ifdef HYPRE_USE_CUDA_HYB
    //printf("Preparing to convert to hyb format\n");
    cusparseStatus_t ct;
    ct=cusparseCreateHybMat(&(A->hybA));
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Creation of A->hybA failed ");
      cudaDeviceReset();
      return 1;
    } //else printf("Creation of A-<hybA succeeded\n");
    
    ct=cusparseDcsr2hyb(A->handle,A->num_rows,A->num_cols,A->descr,A->data_device, A->i_device, A->j_device,
			A->hybA,10,
			CUSPARSE_HYB_PARTITION_AUTO);
    cudaDeviceSynchronize();
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Conversion to hybrid format failed ");
      cudaDeviceReset();
      return 1;
    }// else printf("Conversion to hybrid format succeeded\n");
#endif
    // WARNING:: assumes that A is static and doesnot change 
  }
  hypre_VectorH2D(x);
  hypre_VectorH2D(y);
  cusparseStatus_t status;
#ifndef HYPRE_USE_CUDA_HYB
  status= cusparseDcsrmv(hypre_CSRMatrixHandle(A) ,
			 CUSPARSE_OPERATION_NON_TRANSPOSE, 
			 A->num_rows, A->num_cols, A->num_nonzeros,
  			 &alpha, hypre_CSRMatrixDescr(A),
			 hypre_CSRMatrixDataDevice(A) ,hypre_CSRMatrixIDevice(A),hypre_CSRMatrixJDevice(A),
  			 hypre_VectorDataDevice(x), &beta, hypre_VectorDataDevice(y));
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
  
#else
    status= cusparseDhybmv(A->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
  			 &alpha, A->descr, A->hybA, 
  			 &x->data_device[0], &beta, &y->data_device[0]);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication using cusparseDhybmv failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
#endif
  cudaDeviceSynchronize();
  //HYPRE_Complex prenorm = hypre_VectorNorm(y);
  hypre_VectorD2H(y);
  //printf("Pre & Post Norm of solution is %lf -> %lf\n",prenorm,hypre_VectorNorm(y));
  POP_RANGE;
}

HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceDevice( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{


   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   //if (y!=b) printf("Ye Olde Matvec call %d\n",offset);
   //else printf("OFFSET Matvec call %d\n",offset);
   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   //printf("Entre hypre_CSRMatrixMatvecOutofPlace CUDA Version %d %d %d\n",A->num_rows,A->num_nonzeros,A->i[A->num_rows]);
   //printf("Size of data varbls is %d Alpha = %lf, beta = %lf \n",sizeof(HYPRE_Complex),alpha,beta);
   if (!(hypre_CSRMatrixDevice(A)))hypre_CSRMatrixMapToDevice(A);
   if (!hypre_VectorDevice(x)) hypre_VectorMapToDevice(x);
   if (!hypre_VectorDevice(b)) hypre_VectorMapToDevice(b);
  // printf("IN CUDAFIED hypre_CSRMatrixMatvec\n");
  
   if (!hypre_CSRMatrixCopiedToDevice(A)){
    hypre_CSRMatrixH2D(A);
    hypre_CSRMatrixCopiedToDevice(A)=1;
#ifdef HYPRE_USE_CUDA_HYB
    //printf("Preparing to convert to hyb format\n");
    cusparseStatus_t ct;
    ct=cusparseCreateHybMat(&(A->hybA));
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Creation of A->hybA failed ");
      cudaDeviceReset();
      return 1;
    } //else printf("Creation of A-<hybA succeeded\n");
    
    ct=cusparseDcsr2hyb(A->handle,A->num_rows,A->num_cols,A->descr,A->data_device, A->i_device, A->j_device,
			A->hybA,10,
			CUSPARSE_HYB_PARTITION_AUTO);
    cudaDeviceSynchronize();
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Conversion to hybrid format failed ");
      cudaDeviceReset();
      return 1;
    }// else printf("Conversion to hybrid format succeeded\n");
#endif
    // WARNING:: assumes that A is static and doesnot change 
  }
  hypre_VectorH2D(x);
  hypre_VectorH2D(b);
  cusparseStatus_t status;
#ifndef HYPRE_USE_CUDA_HYB
  status= cusparseDcsrmv(hypre_CSRMatrixHandle(A) ,
			 CUSPARSE_OPERATION_NON_TRANSPOSE, 
			 A->num_rows-offset, A->num_cols, A->num_nonzeros,
  			 &alpha, hypre_CSRMatrixDescr(A),
			 hypre_CSRMatrixDataDevice(A) ,hypre_CSRMatrixIDevice(A)+offset,hypre_CSRMatrixJDevice(A),
  			 hypre_VectorDataDevice(x), &beta, hypre_VectorDataDevice(b)+offset);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
  
#else
    status= cusparseDhybmv(A->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
  			 &alpha, A->descr, A->hybA, 
  			 &x->data_device[0], &beta, &y->data_device[0]);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication using cusparseDhybmv failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
#endif
  cudaDeviceSynchronize();
  //HYPRE_Complex prenorm = hypre_VectorNorm(y);
  hypre_VectorD2HCross(y,b,offset,b_size);
  //printf("Pre & Post Norm of solution is %lf -> %lf\n",prenorm,hypre_VectorNorm(y));
  return ierr;
}

HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceDeviceAsync( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset     )
{


   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   //if (y!=b) printf("Ye Olde Matvec call %d\n",offset);
   //else printf("OFFSET Matvec call %d\n",offset);
   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   //printf("Entre hypre_CSRMatrixMatvecOutofPlace CUDA Version %d %d %d\n",A->num_rows,A->num_nonzeros,A->i[A->num_rows]);
   //printf("Size of data varbls is %d Alpha = %lf, beta = %lf \n",sizeof(HYPRE_Complex),alpha,beta);
   if (!(hypre_CSRMatrixDevice(A)))hypre_CSRMatrixMapToDevice(A);
   if (!hypre_VectorDevice(x)) hypre_VectorMapToDevice(x);
   if (!hypre_VectorDevice(b)) hypre_VectorMapToDevice(b);
  // printf("IN CUDAFIED hypre_CSRMatrixMatvec\n");
  
   if (!hypre_CSRMatrixCopiedToDevice(A)){
    hypre_CSRMatrixH2D(A);
    hypre_CSRMatrixCopiedToDevice(A)=1;
#ifdef HYPRE_USE_CUDA_HYB
    //printf("Preparing to convert to hyb format\n");
    cusparseStatus_t ct;
    ct=cusparseCreateHybMat(&(A->hybA));
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Creation of A->hybA failed ");
      cudaDeviceReset();
      return 1;
    } //else printf("Creation of A-<hybA succeeded\n");
    
    ct=cusparseDcsr2hyb(A->handle,A->num_rows,A->num_cols,A->descr,A->data_device, A->i_device, A->j_device,
			A->hybA,10,
			CUSPARSE_HYB_PARTITION_AUTO);
    cudaDeviceSynchronize();
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Conversion to hybrid format failed ");
      cudaDeviceReset();
      return 1;
    }// else printf("Conversion to hybrid format succeeded\n");
#endif
    // WARNING:: assumes that A is static and doesnot change 
  }
  hypre_VectorH2D(x);
  hypre_VectorH2D(b);
  cusparseStatus_t status;
#ifndef HYPRE_USE_CUDA_HYB
  status= cusparseDcsrmv(hypre_CSRMatrixHandle(A) ,
			 CUSPARSE_OPERATION_NON_TRANSPOSE, 
			 A->num_rows-offset, A->num_cols, A->num_nonzeros,
  			 &alpha, hypre_CSRMatrixDescr(A),
			 hypre_CSRMatrixDataDevice(A) ,hypre_CSRMatrixIDevice(A)+offset,hypre_CSRMatrixJDevice(A),
  			 hypre_VectorDataDevice(x), &beta, hypre_VectorDataDevice(b)+offset);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
  
#else
    status= cusparseDhybmv(A->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
  			 &alpha, A->descr, A->hybA, 
  			 &x->data_device[0], &beta, &y->data_device[0]);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication using cusparseDhybmv failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
#endif
  cudaDeviceSynchronize();
  //HYPRE_Complex prenorm = hypre_VectorNorm(y);
  hypre_VectorD2HCross(y,b,offset,b_size);
  //printf("Pre & Post Norm of solution is %lf -> %lf\n",prenorm,hypre_VectorNorm(y));
  return ierr;
}

HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceHybrid( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset1     )
{
  /* This version computes the top section of the product on the host and 
     sends the bottom half to the device.
  */
  PUSH_RANGE("MV-Hybrid",4);
  HYPRE_Int  offset,offset2; // Total and 2nd offset for the hybrid operation

  offset2=hypre_VectorSize(x)*0.43; // needs to take offset1 into account. PBUGS
  offset=offset2;

   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   static cudaStream_t s;
   static int first_call=0;
   if (!first_call){
     first_call=1;
     gpuErrchk4(cudaStreamCreate(&s));
   }

   
     
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   //if (y!=b) printf("Ye Olde Matvec call %d\n",offset);
   //else printf("OFFSET Matvec call %d\n",offset);
   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   //printf("Entre hypre_CSRMatrixMatvecOutofPlace CUDA Version %d %d %d\n",A->num_rows,A->num_nonzeros,A->i[A->num_rows]);
   //printf("Size of data varbls is %d Alpha = %lf, beta = %lf \n",sizeof(HYPRE_Complex),alpha,beta);
   if (!(hypre_CSRMatrixDevice(A)))hypre_CSRMatrixMapToDevice(A);
   if (!hypre_VectorDevice(x)) hypre_VectorMapToDevice(x);
   if (!hypre_VectorDevice(b)) hypre_VectorMapToDevice(b);
  // printf("IN CUDAFIED hypre_CSRMatrixMatvec\n");
  
   if (!hypre_CSRMatrixCopiedToDevice(A)){
     hypre_CSRMatrixH2DAsync(A,s);
     hypre_CSRMatrixCopiedToDevice(A)=1;
#ifdef HYPRE_USE_CUDA_HYB
    //printf("Preparing to convert to hyb format\n");
    cusparseStatus_t ct;
    ct=cusparseCreateHybMat(&(A->hybA));
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Creation of A->hybA failed ");
      cudaDeviceReset();
      return 1;
    } //else printf("Creation of A-<hybA succeeded\n");
    
    ct=cusparseDcsr2hyb(A->handle,A->num_rows,A->num_cols,A->descr,A->data_device, A->i_device, A->j_device,
			A->hybA,10,
			CUSPARSE_HYB_PARTITION_AUTO);
    cudaDeviceSynchronize();
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Conversion to hybrid format failed ");
      cudaDeviceReset();
      return 1;
    }// else printf("Conversion to hybrid format succeeded\n");
#endif
    // WARNING:: assumes that A is static and doesnot change 
  }
   hypre_VectorH2DAsync(x,s);
   hypre_VectorH2DAsync(b,s);
  cusparseStatus_t status;
#ifndef HYPRE_USE_CUDA_HYB
  status=cusparseSetStream(hypre_CSRMatrixHandle(A) ,s);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR:: cusparseSetStream Failed\n");
  } 
  status= cusparseDcsrmv(hypre_CSRMatrixHandle(A) ,
			 CUSPARSE_OPERATION_NON_TRANSPOSE, 
			 A->num_rows-offset, A->num_cols, A->num_nonzeros,
  			 &alpha, hypre_CSRMatrixDescr(A),
			 hypre_CSRMatrixDataDevice(A) ,hypre_CSRMatrixIDevice(A)+offset,hypre_CSRMatrixJDevice(A),
  			 hypre_VectorDataDevice(x), &beta, hypre_VectorDataDevice(b)+offset);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed");
    cudaDeviceReset();
    return 1;
  } //else //printf("SXS SpMV done\n");
  
#else
    status= cusparseDhybmv(A->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
  			 &alpha, A->descr, A->hybA, 
  			 &x->data_device[0], &beta, &y->data_device[0]);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication using cusparseDhybmv failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
#endif

  //HYPRE_Complex prenorm = hypre_VectorNorm(y);
  hypre_VectorD2HCrossAsync(y,b,offset,b_size,s);
  //printf("Pre & Post Norm of solution is %lf -> %lf\n",prenorm,hypre_VectorNorm(y));

  // Call the host code
  hypre_CSRMatrixMatvecStrip( alpha,A,x,beta,b,y,offset1,offset2);
			    
  cudaDeviceSynchronize();

  POP_RANGE
  return ierr;
}
HYPRE_Int
hypre_CSRMatrixMatvecStrip( HYPRE_Complex    alpha,
			    hypre_CSRMatrix *A,
			    hypre_Vector    *x,
			    HYPRE_Complex    beta,
			    hypre_Vector    *b,
			    hypre_Vector    *y,HYPRE_Int start, HYPRE_Int end     )
{
  PUSH_RANGE("Matvec Strip",5);
  HYPRE_Complex    *A_data;
  HYPRE_Int        *A_i; 
  HYPRE_Int        *A_j; 
  HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
  HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   
  HYPRE_Complex    *x_data;
  HYPRE_Complex    *y_data;
  HYPRE_Complex    *b_data;
  HYPRE_Int         x_size = hypre_VectorSize(x);
  HYPRE_Int         y_size = hypre_VectorSize(y);
  HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);

  
  HYPRE_Complex     bba;
  

  
  HYPRE_Int         ierr = 0;
 
  
  HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(A);;
  
  
  hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
  
  //printf("ENTERED CSR_MATVEC %lf,%lf NUM_VECS = %d \n",alpha,beta,num_vectors);
  
  if (num_cols != x_size)
    ierr = 1;
  
  if (num_rows != y_size)
    ierr = 2;
  
  if (num_cols != x_size && num_rows != y_size)
    ierr = 3;
  
  
  // Copy data: over
  
  x_data = hypre_VectorData(x);
  y_data = hypre_VectorData(y);
  b_data = hypre_VectorData(b);
  A_data   = hypre_CSRMatrixData(A);
  A_i      = hypre_CSRMatrixI(A);
  A_j      = hypre_CSRMatrixJ(A);
  bba = beta / alpha;
  int static count=0;
  //printf("ENTERING DATA REGION %d %p size = %d\n",++count,A_data,nnz);
  int i, jj;
  {
  if (alpha == 0.0)
    {
      //std::cout<<"csr_matvec RAJA alpha=0.0\n";
      printf("MATVEC ALPHA =0 case\n");

      for(i=start;i<end;i++){
	y_data[i] = beta*b_data[i];
      }
      return ierr;
    }
  

  /*-----------------------------------------------------------------
   * y += A*x
   *-----------------------------------------------------------------*/
  for(i=start;i<end;i++)
    {
      y_data[i] = bba*b_data[i];
      HYPRE_Complex temp = 0.0;
      for (jj = A_i[i]; jj < A_i[i+1]; jj++)
	temp += A_data[jj] * x_data[A_j[jj]];
      y_data[i] += temp;
      y_data[i] *= alpha;
    }
  }
  POP_RANGE;
  return ierr;
}
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceHybrid2( HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 HYPRE_Int        offset1     )
{
  /* This version computes the bottom section of the product on the host and 
     sends the top half to the device.
  */
  PUSH_RANGE("MV-Hybrid2",4);
  HYPRE_Int  offset,offset2; // Total and 2nd offset for the hybrid operation
  float fraction=0.6000;
  offset2=hypre_VectorSize(x)*fraction; // needs to take offset1 into account. PBUGS
  offset=offset2;
  y->ref_count++;
  //printf("Call to Hybrid:: ref_count = %d\n",y->ref_count);
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
   /*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y);
   HYPRE_Int         x_size = hypre_VectorSize(x);
   HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   /*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
   HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   HYPRE_Complex     temp, tempx;

   HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   HYPRE_Real        xpar=0.7;

   HYPRE_Int         ierr = 0;
   hypre_Vector	    *x_tmp = NULL;

   static cudaStream_t s;
   static int first_call=0;
   if (!first_call){
     first_call=1;
     gpuErrchk4(cudaStreamCreate(&s));
   }

   
     
   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
   //if (y!=b) printf("Ye Olde Matvec call %d\n",offset);
   //else printf("OFFSET Matvec call %d\n",offset);
   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
   hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

   if (num_cols != x_size)
      ierr = 1;

   if (num_rows != y_size || num_rows != b_size)
      ierr = 2;

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
      ierr = 3;

   //printf("Entre hypre_CSRMatrixMatvecOutofPlace CUDA Version %d %d oFFSET %d\n",A->num_rows,A->num_nonzeros,offset2);
   //printf("Size of data varbls is %d Alpha = %lf, beta = %lf \n",sizeof(HYPRE_Complex),alpha,beta);
   if (!(hypre_CSRMatrixDevice(A)))hypre_CSRMatrixMapToDevice(A);
   if (!hypre_VectorDevice(x)) hypre_VectorMapToDevice(x);
   if (!hypre_VectorDevice(b)) hypre_VectorMapToDevice(b);
  // printf("IN CUDAFIED hypre_CSRMatrixMatvec\n");
   b->dev->offset1=offset1;
   b->dev->offset2=offset2;
   y->dev->offset1=offset1;
   y->dev->offset2=offset2;
   x->dev->offset1=offset1;
   x->dev->offset2=offset2;
   if (!hypre_CSRMatrixCopiedToDevice(A)){
     hypre_CSRMatrixH2DAsyncPartial(A,fraction,s);
     hypre_CSRMatrixCopiedToDevice(A)=1;
#ifdef HYPRE_USE_CUDA_HYB
    //printf("Preparing to convert to hyb format\n");
    cusparseStatus_t ct;
    ct=cusparseCreateHybMat(&(A->hybA));
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Creation of A->hybA failed ");
      cudaDeviceReset();
      return 1;
    } //else printf("Creation of A-<hybA succeeded\n");
    
    ct=cusparseDcsr2hyb(A->handle,A->num_rows,A->num_cols,A->descr,A->data_device, A->i_device, A->j_device,
			A->hybA,10,
			CUSPARSE_HYB_PARTITION_AUTO);
    cudaDeviceSynchronize();
    if (ct != CUSPARSE_STATUS_SUCCESS) {
      printf("Conversion to hybrid format failed ");
      cudaDeviceReset();
      return 1;
    }// else printf("Conversion to hybrid format succeeded\n");
#endif
    // WARNING:: assumes that A is static and doesnot change 
  }
   hypre_VectorH2DAsync(x,s);
   hypre_VectorH2DAsyncPartial(b,offset2-offset1,s); // PBUGS if offset1 !=0
   //hypre_VectorH2DAsync(b,s);
  cusparseStatus_t status;
#ifndef HYPRE_USE_CUDA_HYB
  status=cusparseSetStream(hypre_CSRMatrixHandle(A) ,s);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR:: cusparseSetStream Failed\n");
  } 
  status= cusparseDcsrmv(hypre_CSRMatrixHandle(A) ,
			 CUSPARSE_OPERATION_NON_TRANSPOSE, 
			 offset2-offset1, A->num_cols, A->num_nonzeros,
  			 &alpha, hypre_CSRMatrixDescr(A),
			 hypre_CSRMatrixDataDevice(A) ,hypre_CSRMatrixIDevice(A)+offset1,hypre_CSRMatrixJDevice(A),
  			 hypre_VectorDataDevice(x), &beta, hypre_VectorDataDevice(b)+offset1);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication failed");
    cudaDeviceReset();
    return 1;
  } //else //printf("SXS SpMV done\n");
  
#else
    status= cusparseDhybmv(A->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
  			 &alpha, A->descr, A->hybA, 
  			 &x->data_device[0], &beta, &y->data_device[0]);
  
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("Matrix-vector multiplication using cusparseDhybmv failed");
    cudaDeviceReset();
    return 1;
  } else //printf("SXS SpMV done\n");
#endif

  //HYPRE_Complex prenorm = hypre_VectorNorm(y);
  hypre_VectorD2HCrossAsync(y,b,offset1,offset2-offset1,s);
  //printf("Pre & Post Norm of solution is %lf -> %lf\n",prenorm,hypre_VectorNorm(y));

  // Call the host code
  hypre_CSRMatrixMatvecStrip( alpha,A,x,beta,b,y,offset2,A->num_rows);
			    
  cudaDeviceSynchronize();

  POP_RANGE
  return ierr;
}
#endif
