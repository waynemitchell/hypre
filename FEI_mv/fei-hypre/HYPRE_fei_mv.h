/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 **********************************************************************EHEADER*/

#ifndef HYPRE_FE_MV_HEADER
#define HYPRE_FE_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI System Interface
 *
 * This interface represents a FE conceptual view of a
 * linear system.  
 *
 * @memo A FE conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Mesh
 **/
/*@{*/

struct hypre_FEMesh_struct;
typedef struct hypre_FEMesh_struct *HYPRE_FEMesh;

/**
 * Create a FE Mesh object.  
 **/

int HYPRE_FEMeshCreate(MPI_Comm comm, HYPRE_FEMesh *mesh);

/**
 * Destroy an FE Mesh object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  
 **/

int HYPRE_FEMeshDestroy(HYPRE_FEMesh mesh);

/**
 * load an FE object
 **/

int HYPRE_FEMeshSetFEObject(HYPRE_FEMesh mesh, void *, void *);

/**
 * initialize all fields in the finite element mesh
 **/

int HYPRE_FEMeshInitFields(HYPRE_FEMesh mesh, int numFields, 
                           int *fieldSizes, int *fieldIDs);

/**
 * initialize an element block
 **/

int HYPRE_FEMeshInitElemBlock(HYPRE_FEMesh mesh, int blockID, int nElements,
                int numNodesPerElement, int *numFieldsPerNode,
                int **nodalFieldIDs, int numElemDOFFieldsPerElement,
                int *elemDOFFieldIDs, int interleaveStrategy);

/**
 * initialize the connectivity of a given element
 **/

int HYPRE_FEMeshInitElem(HYPRE_FEMesh mesh, int blockID, int elemID,
                         int *elemConn);

/**
 * initialize the shared nodes between processors
 **/

int HYPRE_FEMeshInitSharedNodes(HYPRE_FEMesh mesh, int nShared,
                                int *sharedIDs, int *sharedLeng,
                                int **sharedProcs);

/**
 * initialization complete
 **/

int HYPRE_FEMeshInitComplete(HYPRE_FEMesh mesh);

/**
 * load node boundary conditions
 **/

int HYPRE_FEMeshLoadNodeBCs(HYPRE_FEMesh mesh, int numNodes,
                            int *nodeIDs, int fieldID, double **alpha,
                            double **beta, double **gamma);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Matrices
 **/
/*@{*/

struct hypre_FEMatrix_struct;
/**
 * The matrix object
 **/
typedef struct hypre_FEMatrix_struct *HYPRE_FEMatrix;

/**
 * create a new FE matrix
 **/

int HYPRE_FEMatrixCreate(MPI_Comm comm, HYPRE_FEMesh mesh, 
                         HYPRE_FEMatrix *matrix);

/**
 * destroy a new FE matrix
 **/

int HYPRE_FEMatrixDestroy(HYPRE_FEMatrix matrix);
   
/**
 * prepare a matrix object for setting coefficient values
 **/

int HYPRE_FEMatrixInitialize(HYPRE_FEMatrix matrix);
   
/**
 * signal that loading has been completed
 **/

int HYPRE_FEMatrixAssemble(HYPRE_FEMatrix matrix);
   
/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR} (default).
 *
 **/
int HYPRE_FEMatrixSetObjectType(HYPRE_FEMatrix  matrix, int type);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see HYPRE_FEMatrixSetObjectType
 **/
int HYPRE_FEMatrixGetObject(HYPRE_FEMatrix matrix, void **object);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FE Vectors
 **/
/*@{*/

struct hypre_FEVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_FEVector_struct *HYPRE_FEVector;

/**
 * Create a vector object.
 **/
int HYPRE_FEVectorCreate(MPI_Comm comm, HYPRE_FEMesh mesh,
                         HYPRE_FEVector  *vector);

/**
 * Destroy a vector object.
 **/
int HYPRE_FEVectorDestroy(HYPRE_FEVector vector);

/**
 * Prepare a vector object for setting coefficient values.
 **/
int HYPRE_FEVectorInitialize(HYPRE_FEVector vector);


/**
 * Finalize the construction of the vector before using.
 **/
int HYPRE_FEVectorAssemble(HYPRE_FEVector vector);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR} (default).
 **/
int HYPRE_FEVectorSetObjectType(HYPRE_FEVector vector, int type);

/**
 * Get a reference to the constructed vector object.
 **/
int HYPRE_FEVectorGetObject(HYPRE_FEVector vector, void **object);

/*@}*/
/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

#ifdef __cplusplus
}
#endif

#endif
