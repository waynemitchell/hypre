/*
 * File:          Hypre_PreconditionedSolver_IOR.c
 * Symbol:        Hypre.PreconditionedSolver-v0.1.7
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.PreconditionedSolver
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 766
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_PreconditionedSolver_IOR.h"

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

static struct Hypre_PreconditionedSolver__epv s_rem__hypre_preconditionedsolver;

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_PreconditionedSolver__cast(
  void* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_PreconditionedSolver__delete(
  void* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_PreconditionedSolver_addRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_PreconditionedSolver_deleteRef(
  void* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_PreconditionedSolver_isSame(
  void* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_PreconditionedSolver_queryInt(
  void* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_PreconditionedSolver_isType(
  void* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetCommunicator(
  void* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetIntParameter(
  void* self,
  const char* name,
  int32_t value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetDoubleParameter(
  void* self,
  const char* name,
  double value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetStringParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetStringParameter(
  void* self,
  const char* name,
  const char* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetIntArrayParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetIntArrayParameter(
  void* self,
  const char* name,
  struct SIDL_int__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetDoubleArrayParameter
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetDoubleArrayParameter(
  void* self,
  const char* name,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetIntValue
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetIntValue(
  void* self,
  const char* name,
  int32_t* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetDoubleValue
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetDoubleValue(
  void* self,
  const char* name,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Setup
 */

static int32_t
remote_Hypre_PreconditionedSolver_Setup(
  void* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Apply
 */

static int32_t
remote_Hypre_PreconditionedSolver_Apply(
  void* self,
  struct Hypre_Vector__object* b,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetOperator
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetOperator(
  void* self,
  struct Hypre_Operator__object* A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetTolerance
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetTolerance(
  void* self,
  double tolerance)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetMaxIterations
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetMaxIterations(
  void* self,
  int32_t max_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetLogging
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetLogging(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPrintLevel
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetPrintLevel(
  void* self,
  int32_t level)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetNumIterations
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetNumIterations(
  void* self,
  int32_t* num_iterations)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetRelResidualNorm
 */

static int32_t
remote_Hypre_PreconditionedSolver_GetRelResidualNorm(
  void* self,
  double* norm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPreconditioner
 */

static int32_t
remote_Hypre_PreconditionedSolver_SetPreconditioner(
  void* self,
  struct Hypre_Solver__object* s)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_PreconditionedSolver__init_remote_epv(void)
{
  struct Hypre_PreconditionedSolver__epv* epv = 
    &s_rem__hypre_preconditionedsolver;

  epv->f__cast                   = remote_Hypre_PreconditionedSolver__cast;
  epv->f__delete                 = remote_Hypre_PreconditionedSolver__delete;
  epv->f_addRef                  = remote_Hypre_PreconditionedSolver_addRef;
  epv->f_deleteRef               = remote_Hypre_PreconditionedSolver_deleteRef;
  epv->f_isSame                  = remote_Hypre_PreconditionedSolver_isSame;
  epv->f_queryInt                = remote_Hypre_PreconditionedSolver_queryInt;
  epv->f_isType                  = remote_Hypre_PreconditionedSolver_isType;
  epv->f_SetCommunicator         = 
    remote_Hypre_PreconditionedSolver_SetCommunicator;
  epv->f_SetIntParameter         = 
    remote_Hypre_PreconditionedSolver_SetIntParameter;
  epv->f_SetDoubleParameter      = 
    remote_Hypre_PreconditionedSolver_SetDoubleParameter;
  epv->f_SetStringParameter      = 
    remote_Hypre_PreconditionedSolver_SetStringParameter;
  epv->f_SetIntArrayParameter    = 
    remote_Hypre_PreconditionedSolver_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    remote_Hypre_PreconditionedSolver_SetDoubleArrayParameter;
  epv->f_GetIntValue             = 
    remote_Hypre_PreconditionedSolver_GetIntValue;
  epv->f_GetDoubleValue          = 
    remote_Hypre_PreconditionedSolver_GetDoubleValue;
  epv->f_Setup                   = remote_Hypre_PreconditionedSolver_Setup;
  epv->f_Apply                   = remote_Hypre_PreconditionedSolver_Apply;
  epv->f_SetOperator             = 
    remote_Hypre_PreconditionedSolver_SetOperator;
  epv->f_SetTolerance            = 
    remote_Hypre_PreconditionedSolver_SetTolerance;
  epv->f_SetMaxIterations        = 
    remote_Hypre_PreconditionedSolver_SetMaxIterations;
  epv->f_SetLogging              = remote_Hypre_PreconditionedSolver_SetLogging;
  epv->f_SetPrintLevel           = 
    remote_Hypre_PreconditionedSolver_SetPrintLevel;
  epv->f_GetNumIterations        = 
    remote_Hypre_PreconditionedSolver_GetNumIterations;
  epv->f_GetRelResidualNorm      = 
    remote_Hypre_PreconditionedSolver_GetRelResidualNorm;
  epv->f_SetPreconditioner       = 
    remote_Hypre_PreconditionedSolver_SetPreconditioner;
  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_PreconditionedSolver__object*
Hypre_PreconditionedSolver__remote(const char *url)
{
  struct Hypre_PreconditionedSolver__object* self =
    (struct Hypre_PreconditionedSolver__object*) malloc(
      sizeof(struct Hypre_PreconditionedSolver__object));

  if (!s_remote_initialized) {
    Hypre_PreconditionedSolver__init_remote_epv();
  }

  self->d_epv    = &s_rem__hypre_preconditionedsolver;
  self->d_object = NULL; /* FIXME */

  return self;
}
