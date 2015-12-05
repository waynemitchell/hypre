/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Header info for the hypre_SStructGraph structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_GRAPH_HEADER
#define hypre_SSTRUCT_GRAPH_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructGraph:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int           part;
   hypre_Index   index;
   int           var;
   int           to_part;     
   hypre_Index   to_index;
   int           to_var;

} hypre_SStructGraphEntry;



typedef struct
{
   int           to_part;
   hypre_Index   to_index;
   int           to_var;
   int           to_boxnum;      /* local box number */
   int           to_proc;
   int           rank;

} hypre_SStructUEntry;

typedef struct
{
   int                  part;
   hypre_Index          index;
   int                  var;
   int                  boxnum;  /* local box number */
   int                  nUentries;
   hypre_SStructUEntry *Uentries;

} hypre_SStructUVEntry;

typedef struct hypre_SStructGraph_struct
{
   MPI_Comm                comm;
   int                     ndim;
   hypre_SStructGrid      *grid;
   hypre_SStructGrid      *domain_grid; /* same as grid by default */
   int                     nparts;
   hypre_SStructPGrid    **pgrids;
   hypre_SStructStencil ***stencils; /* each (part, var) has a stencil */

   /* info for fem-based user input */
   int                    *fem_nsparse;
   int                   **fem_sparse_i;
   int                   **fem_sparse_j;
   int                   **fem_entries;

   /* U-graph info: Entries are referenced via local grid-variable rank. */
   int                     nUventries;  /* number of iUventries */
   int                     aUventries;  /* alloc size of iUventries */
   int                    *iUventries;

   hypre_SStructUVEntry  **Uventries;
   int                     totUentries;

   int                     ref_count;

   int                     type;    /* GEC0203 */

   hypre_SStructGraphEntry **graph_entries; /* these are stored from
                                             * the AddGraphEntries calls
                                             * and then deleted in the
                                             * GraphAssemble */
   int                     n_graph_entries; /* number graph entries */
   int                     a_graph_entries; /* alloced graph entries */
   


} hypre_SStructGraph;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGraph
 *--------------------------------------------------------------------------*/

#define hypre_SStructGraphComm(graph)           ((graph) -> comm)
#define hypre_SStructGraphNDim(graph)           ((graph) -> ndim)
#define hypre_SStructGraphGrid(graph)           ((graph) -> grid)
#define hypre_SStructGraphDomainGrid(graph)     ((graph) -> domain_grid)
#define hypre_SStructGraphNParts(graph)         ((graph) -> nparts)
#define hypre_SStructGraphPGrids(graph) \
   hypre_SStructGridPGrids(hypre_SStructGraphGrid(graph))
#define hypre_SStructGraphPGrid(graph, p) \
   hypre_SStructGridPGrid(hypre_SStructGraphGrid(graph), p)
#define hypre_SStructGraphStencils(graph)       ((graph) -> stencils)
#define hypre_SStructGraphStencil(graph, p, v)  ((graph) -> stencils[p][v])

#define hypre_SStructGraphFEMNSparse(graph)     ((graph) -> fem_nsparse)
#define hypre_SStructGraphFEMSparseI(graph)     ((graph) -> fem_sparse_i)
#define hypre_SStructGraphFEMSparseJ(graph)     ((graph) -> fem_sparse_j)
#define hypre_SStructGraphFEMEntries(graph)     ((graph) -> fem_entries)
#define hypre_SStructGraphFEMPNSparse(graph, p) ((graph) -> fem_nsparse[p])
#define hypre_SStructGraphFEMPSparseI(graph, p) ((graph) -> fem_sparse_i[p])
#define hypre_SStructGraphFEMPSparseJ(graph, p) ((graph) -> fem_sparse_j[p])
#define hypre_SStructGraphFEMPEntries(graph, p) ((graph) -> fem_entries[p])

#define hypre_SStructGraphNUVEntries(graph)     ((graph) -> nUventries)
#define hypre_SStructGraphAUVEntries(graph)     ((graph) -> aUventries)
#define hypre_SStructGraphIUVEntries(graph)     ((graph) -> iUventries)
#define hypre_SStructGraphIUVEntry(graph, i)    ((graph) -> iUventries[i])
#define hypre_SStructGraphUVEntries(graph)      ((graph) -> Uventries)
#define hypre_SStructGraphUVEntry(graph, i)     ((graph) -> Uventries[i])
#define hypre_SStructGraphTotUEntries(graph)    ((graph) -> totUentries)
#define hypre_SStructGraphRefCount(graph)       ((graph) -> ref_count)
#define hypre_SStructGraphObjectType(graph)     ((graph) -> type)
#define hypre_SStructGraphEntries(graph)        ((graph) -> graph_entries)
#define hypre_SStructNGraphEntries(graph)       ((graph) -> n_graph_entries)
#define hypre_SStructAGraphEntries(graph)       ((graph) -> a_graph_entries)


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUVEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUVEntryPart(Uv)        ((Uv) -> part)
#define hypre_SStructUVEntryIndex(Uv)       ((Uv) -> index)
#define hypre_SStructUVEntryVar(Uv)         ((Uv) -> var)
#define hypre_SStructUVEntryBoxnum(Uv)      ((Uv) -> boxnum)
#define hypre_SStructUVEntryNUEntries(Uv)   ((Uv) -> nUentries)
#define hypre_SStructUVEntryUEntries(Uv)    ((Uv) -> Uentries)
#define hypre_SStructUVEntryUEntry(Uv, i)  &((Uv) -> Uentries[i])
#define hypre_SStructUVEntryToPart(Uv, i)   ((Uv) -> Uentries[i].to_part)
#define hypre_SStructUVEntryToIndex(Uv, i)  ((Uv) -> Uentries[i].to_index)
#define hypre_SStructUVEntryToVar(Uv, i)    ((Uv) -> Uentries[i].to_var)
#define hypre_SStructUVEntryToBoxnum(Uv, i) ((Uv) -> Uentries[i].to_boxnum)
#define hypre_SStructUVEntryToProc(Uv, i)   ((Uv) -> Uentries[i].to_proc)
#define hypre_SStructUVEntryRank(Uv, i)     ((Uv) -> Uentries[i].rank)
/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructUEntry
 *--------------------------------------------------------------------------*/

#define hypre_SStructUEntryToPart(U)   ((U) -> to_part)
#define hypre_SStructUEntryToIndex(U)  ((U) -> to_index)
#define hypre_SStructUEntryToVar(U)    ((U) -> to_var)
#define hypre_SStructUEntryToBoxnum(U) ((U) -> to_boxnum)
#define hypre_SStructUEntryToProc(U)   ((U) -> to_proc)
#define hypre_SStructUEntryRank(U)     ((U) -> rank)


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructGraphEntry
 *--------------------------------------------------------------------------*/
#define hypre_SStructGraphEntryPart(g)     ((g) -> part)
#define hypre_SStructGraphEntryIndex(g)    ((g) -> index)
#define hypre_SStructGraphEntryVar(g)      ((g) -> var)
#define hypre_SStructGraphEntryToPart(g)   ((g) -> to_part)
#define hypre_SStructGraphEntryToIndex(g)  ((g) -> to_index)
#define hypre_SStructGraphEntryToVar(g)    ((g) -> to_var)




#endif
