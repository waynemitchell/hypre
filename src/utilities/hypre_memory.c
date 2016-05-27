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
 * Memory management utilities
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
#endif

/******************************************************************************
 *
 * Standard routines
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_OutOfMemory( size_t size )
{
   hypre_printf("Out of memory trying to allocate %d bytes\n", (HYPRE_Int) size);
   fflush(stdout);

   hypre_error(HYPRE_ERROR_MEMORY);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_MAlloc( size_t size )
{
   char *ptr;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      ptr = _umalloc_(size);
#else
#ifndef HYPRE_USE_CUDA
      ptr = malloc(size);
#else
      size_t new_size;
	size_t pgz=getpagesize();
	if (size>pgz*1){
	  
	  //new_size=(size/getpagesize()+1)*getpagesize();
	  new_size=((size+pgz-1)/pgz)*pgz;
	  //printf("Pagealigned memalloc %d -> %d , pagesize %d\n",size,new_size,pgz);
	} else new_size=size;
	
	if (posix_memalign((void**)&ptr,getpagesize(),new_size)){
	  printf("ERROR:: allocating page aligned memory in hypre_CAlloc of size %d bytes\n",size);
	} else {
	  //memset(ptr,0,new_size);
	}
#endif
#endif

#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
#endif
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_CAlloc( size_t count,
              size_t elt_size )
{
   char   *ptr;
   size_t  size = count*elt_size;

   if (size > 0)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      ptr = _ucalloc_(count, elt_size);
#else
#ifndef HYPRE_USE_CUDA
      ptr = calloc(count, elt_size);
#else
      if (size<0) {
	ptr = calloc(count, elt_size);
      } else {
	size_t new_size;
	size_t pgz=getpagesize();
	if (size>pgz*1){
	  
	  //new_size=(size/getpagesize()+1)*getpagesize();
	  new_size=((size+pgz-1)/pgz)*pgz;
	  //printf("Pagealigned memalloc %d -> %d , pagesize %d\n",size,new_size,pgz);
	} else new_size=size;
	
	if (posix_memalign((void**)&ptr,getpagesize(),new_size)){
	  printf("ERROR:: allocating page aligned memory in hypre_CAlloc of size %d bytes\n",size);
	} else {
	  memset(ptr,0,new_size);
	}
      }
#endif
#endif

#if 1
      if (ptr == NULL)
      {
        hypre_OutOfMemory(size);
      }
#endif
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_ReAlloc( char   *ptr,
               size_t  size )
{
#ifdef HYPRE_USE_UMALLOC
   if (ptr == NULL)
   {
      ptr = hypre_MAlloc(size);
   }
   else if (size == 0)
   {
      hypre_Free(ptr);
   }
   else
   {
      HYPRE_Int threadid = hypre_GetThreadID();
      ptr = _urealloc_(ptr, size);
   }
#else
   if (ptr == NULL)
   {
      ptr = malloc(size);
   }
   else
   {
      ptr = realloc(ptr, size);
   }
#endif

#if 1
   if ((ptr == NULL) && (size > 0))
   {
      hypre_OutOfMemory(size);
   }
#endif

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void
hypre_Free( char *ptr )
{
   if (ptr)
   {
#ifdef HYPRE_USE_UMALLOC
      HYPRE_Int threadid = hypre_GetThreadID();

      _ufree_(ptr);
#else
      free(ptr);
#endif
   }
}
