/*
 * File:          Hypre_StructBuildVector_IOR.c
 * Symbol:        Hypre.StructBuildVector-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.StructBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 582
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_StructBuildVector_IOR.h"

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variables for managing EPV initialization.
 */

static int s_remote_initialized = 0;

static struct Hypre_StructBuildVector__epv s_rem__hypre_structbuildvector;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_StructBuildVector__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_StructBuildVector__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_StructBuildVector_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_StructBuildVector_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_StructBuildVector_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_StructBuildVector_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_StructBuildVector_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_StructBuildVector_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_StructBuildVector_Initialize(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_StructBuildVector_Assemble(
  void* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_StructBuildVector_GetObject(
  void* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_StructBuildVector_SetGrid(
  void* self,
  struct Hypre_StructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStencil
 */

static int32_t
remote_Hypre_StructBuildVector_SetStencil(
  void* self,
  struct Hypre_StructStencil__object* stencil)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValue
 */

static int32_t
remote_Hypre_StructBuildVector_SetValue(
  void* self,
  struct SIDL_int__array* grid_index,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_StructBuildVector_SetBoxValues(
  void* self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_StructBuildVector__init_remote_epv(void)
{
  struct Hypre_StructBuildVector__epv* epv = &s_rem__hypre_structbuildvector;

  epv->f__cast           = remote_Hypre_StructBuildVector__cast;
  epv->f__delete         = remote_Hypre_StructBuildVector__delete;
  epv->f_addRef          = remote_Hypre_StructBuildVector_addRef;
  epv->f_deleteRef       = remote_Hypre_StructBuildVector_deleteRef;
  epv->f_isSame          = remote_Hypre_StructBuildVector_isSame;
  epv->f_queryInt        = remote_Hypre_StructBuildVector_queryInt;
  epv->f_isType          = remote_Hypre_StructBuildVector_isType;
  epv->f_SetCommunicator = remote_Hypre_StructBuildVector_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_StructBuildVector_Initialize;
  epv->f_Assemble        = remote_Hypre_StructBuildVector_Assemble;
  epv->f_GetObject       = remote_Hypre_StructBuildVector_GetObject;
  epv->f_SetGrid         = remote_Hypre_StructBuildVector_SetGrid;
  epv->f_SetStencil      = remote_Hypre_StructBuildVector_SetStencil;
  epv->f_SetValue        = remote_Hypre_StructBuildVector_SetValue;
  epv->f_SetBoxValues    = remote_Hypre_StructBuildVector_SetBoxValues;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_StructBuildVector__object*
Hypre_StructBuildVector__remote(const char *url)
{
  struct Hypre_StructBuildVector__object* self =
    (struct Hypre_StructBuildVector__object*) malloc(
      sizeof(struct Hypre_StructBuildVector__object));

  if (!s_remote_initialized) {
    Hypre_StructBuildVector__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_structbuildvector;
  self->d_object = NULL; /* FIXME */

  return self;
}
