/*
 * File:          Hypre_SStructParCSRVector_IOR.c
 * Symbol:        Hypre.SStructParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 847
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_SStructParCSRVector_IOR.h"
#ifndef included_SIDL_BaseClass_Impl_h
#include "SIDL_BaseClass_Impl.h"
#endif
#ifndef included_SIDL_BaseClass_h
#include "SIDL_BaseClass.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_SIDL_ClassInfoI_h
#include "SIDL_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 8;
/*
 * Static variable to hold shared ClassInfo interface.
 */

static SIDL_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_remote_initialized = 0;

static struct Hypre_SStructParCSRVector__epv s_new__hypre_sstructparcsrvector;
static struct Hypre_SStructParCSRVector__epv s_rem__hypre_sstructparcsrvector;

static struct Hypre_ProblemDefinition__epv s_new__hypre_problemdefinition;
static struct Hypre_ProblemDefinition__epv s_rem__hypre_problemdefinition;

static struct Hypre_SStructBuildVector__epv s_new__hypre_sstructbuildvector;
static struct Hypre_SStructBuildVector__epv s_rem__hypre_sstructbuildvector;

static struct Hypre_Vector__epv s_new__hypre_vector;
static struct Hypre_Vector__epv s_rem__hypre_vector;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_SStructParCSRVector__set_epv(
  struct Hypre_SStructParCSRVector__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_Hypre_SStructParCSRVector__cast(
  struct Hypre_SStructParCSRVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.SStructParCSRVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "Hypre.ProblemDefinition")) {
    cast = (void*) &s0->d_hypre_problemdefinition;
  } else if (!strcmp(name, "Hypre.SStructBuildVector")) {
    cast = (void*) &s0->d_hypre_sstructbuildvector;
  } else if (!strcmp(name, "Hypre.Vector")) {
    cast = (void*) &s0->d_hypre_vector;
  } else if (!strcmp(name, "SIDL.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "SIDL.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_Hypre_SStructParCSRVector__delete(
  struct Hypre_SStructParCSRVector__object* self)
{
  Hypre_SStructParCSRVector__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_SStructParCSRVector__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_SStructParCSRVector__init_epv(
  struct Hypre_SStructParCSRVector__object* self)
{
  struct Hypre_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  struct Hypre_SStructParCSRVector__epv* epv = 
    &s_new__hypre_sstructparcsrvector;
  struct Hypre_ProblemDefinition__epv*   e0  = &s_new__hypre_problemdefinition;
  struct Hypre_SStructBuildVector__epv*  e1  = &s_new__hypre_sstructbuildvector;
  struct Hypre_Vector__epv*              e2  = &s_new__hypre_vector;
  struct SIDL_BaseClass__epv*            e3  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv*        e4  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast           = ior_Hypre_SStructParCSRVector__cast;
  epv->f__delete         = ior_Hypre_SStructParCSRVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = (void (*)(struct 
    Hypre_SStructParCSRVector__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef       = (void (*)(struct 
    Hypre_SStructParCSRVector__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame          = (SIDL_bool (*)(struct 
    Hypre_SStructParCSRVector__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_SStructParCSRVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType          = (SIDL_bool (*)(struct 
    Hypre_SStructParCSRVector__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo    = (struct SIDL_ClassInfo__object* (*)(struct 
    Hypre_SStructParCSRVector__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator = NULL;
  epv->f_Initialize      = NULL;
  epv->f_Assemble        = NULL;
  epv->f_GetObject       = NULL;
  epv->f_SetGrid         = NULL;
  epv->f_SetValues       = NULL;
  epv->f_SetBoxValues    = NULL;
  epv->f_AddToValues     = NULL;
  epv->f_AddToBoxValues  = NULL;
  epv->f_Gather          = NULL;
  epv->f_GetValues       = NULL;
  epv->f_GetBoxValues    = NULL;
  epv->f_SetComplex      = NULL;
  epv->f_Print           = NULL;
  epv->f_Clear           = NULL;
  epv->f_Copy            = NULL;
  epv->f_Clone           = NULL;
  epv->f_Scale           = NULL;
  epv->f_Dot             = NULL;
  epv->f_Axpy            = NULL;

  Hypre_SStructParCSRVector__set_epv(epv);

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_SStructGrid__object*)) epv->f_SetGrid;
  e1->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e1->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e1->f_Gather          = (int32_t (*)(void*)) epv->f_Gather;
  e1->f_GetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,double*)) epv->f_GetValues;
  e1->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array**)) epv->f_GetBoxValues;
  e1->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e1->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  e2->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete   = (void (*)(void*)) epv->f__delete;
  e2->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy      = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e2->f_Clone     = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e2->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot       = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy      = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e3->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e3->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e3->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e4->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete   = (void (*)(void*)) epv->f__delete;
  e4->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  s_method_initialized = 1;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(SIDL_ClassInfo *info)
{
  if (s_classInfo_init) {
    SIDL_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = SIDL_ClassInfoI__create();
    s_classInfo = SIDL_ClassInfo__cast(impl);
    if (impl) {
      SIDL_ClassInfoI_setName(impl, "Hypre.SStructParCSRVector");
      SIDL_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      SIDL_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    SIDL_ClassInfo_addRef(*info);
  }
}

/*
 * initMetadata: store IOR version & class in SIDL.BaseClass's data
 */

static void
initMetadata(struct Hypre_SStructParCSRVector__object* self)
{
  if (self) {
    struct SIDL_BaseClass__data *data = 
      SIDL_BaseClass__get_data(SIDL_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct Hypre_SStructParCSRVector__object*
Hypre_SStructParCSRVector__new(void)
{
  struct Hypre_SStructParCSRVector__object* self =
    (struct Hypre_SStructParCSRVector__object*) malloc(
      sizeof(struct Hypre_SStructParCSRVector__object));
  Hypre_SStructParCSRVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_SStructParCSRVector__init(
  struct Hypre_SStructParCSRVector__object* self)
{
  struct Hypre_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_SStructParCSRVector__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv  = &s_new__hypre_problemdefinition;
  s0->d_hypre_sstructbuildvector.d_epv = &s_new__hypre_sstructbuildvector;
  s0->d_hypre_vector.d_epv             = &s_new__hypre_vector;
  s0->d_epv                            = &s_new__hypre_sstructparcsrvector;

  s0->d_hypre_problemdefinition.d_object = self;

  s0->d_hypre_sstructbuildvector.d_object = self;

  s0->d_hypre_vector.d_object = self;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_SStructParCSRVector__fini(
  struct Hypre_SStructParCSRVector__object* self)
{
  struct Hypre_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
Hypre_SStructParCSRVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct Hypre_SStructParCSRVector__external
s_externalEntryPoints = {
  Hypre_SStructParCSRVector__new,
  Hypre_SStructParCSRVector__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructParCSRVector__external*
Hypre_SStructParCSRVector__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_SStructParCSRVector__cast(
  struct Hypre_SStructParCSRVector__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_SStructParCSRVector__delete(
  struct Hypre_SStructParCSRVector__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_SStructParCSRVector_addRef(
  struct Hypre_SStructParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_SStructParCSRVector_deleteRef(
  struct Hypre_SStructParCSRVector__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_SStructParCSRVector_isSame(
  struct Hypre_SStructParCSRVector__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_SStructParCSRVector_queryInt(
  struct Hypre_SStructParCSRVector__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_SStructParCSRVector_isType(
  struct Hypre_SStructParCSRVector__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_Hypre_SStructParCSRVector_getClassInfo(
  struct Hypre_SStructParCSRVector__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetCommunicator
 */

static int32_t
remote_Hypre_SStructParCSRVector_SetCommunicator(
  struct Hypre_SStructParCSRVector__object* self,
  void* mpi_comm)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Initialize
 */

static int32_t
remote_Hypre_SStructParCSRVector_Initialize(
  struct Hypre_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Assemble
 */

static int32_t
remote_Hypre_SStructParCSRVector_Assemble(
  struct Hypre_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetObject
 */

static int32_t
remote_Hypre_SStructParCSRVector_GetObject(
  struct Hypre_SStructParCSRVector__object* self,
  struct SIDL_BaseInterface__object** A)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetGrid
 */

static int32_t
remote_Hypre_SStructParCSRVector_SetGrid(
  struct Hypre_SStructParCSRVector__object* self,
  struct Hypre_SStructGrid__object* grid)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_SetValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetBoxValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_SetBoxValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_AddToValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  struct SIDL_double__array* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddToBoxValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_AddToBoxValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array* values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Gather
 */

static int32_t
remote_Hypre_SStructParCSRVector_Gather(
  struct Hypre_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_GetValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  double* value)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:GetBoxValues
 */

static int32_t
remote_Hypre_SStructParCSRVector_GetBoxValues(
  struct Hypre_SStructParCSRVector__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t var,
  struct SIDL_double__array** values)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetComplex
 */

static int32_t
remote_Hypre_SStructParCSRVector_SetComplex(
  struct Hypre_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Print
 */

static int32_t
remote_Hypre_SStructParCSRVector_Print(
  struct Hypre_SStructParCSRVector__object* self,
  const char* filename,
  int32_t all)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clear
 */

static int32_t
remote_Hypre_SStructParCSRVector_Clear(
  struct Hypre_SStructParCSRVector__object* self)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Copy
 */

static int32_t
remote_Hypre_SStructParCSRVector_Copy(
  struct Hypre_SStructParCSRVector__object* self,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Clone
 */

static int32_t
remote_Hypre_SStructParCSRVector_Clone(
  struct Hypre_SStructParCSRVector__object* self,
  struct Hypre_Vector__object** x)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Scale
 */

static int32_t
remote_Hypre_SStructParCSRVector_Scale(
  struct Hypre_SStructParCSRVector__object* self,
  double a)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Dot
 */

static int32_t
remote_Hypre_SStructParCSRVector_Dot(
  struct Hypre_SStructParCSRVector__object* self,
  struct Hypre_Vector__object* x,
  double* d)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:Axpy
 */

static int32_t
remote_Hypre_SStructParCSRVector_Axpy(
  struct Hypre_SStructParCSRVector__object* self,
  double a,
  struct Hypre_Vector__object* x)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_SStructParCSRVector__init_remote_epv(void)
{
  struct Hypre_SStructParCSRVector__epv* epv = 
    &s_rem__hypre_sstructparcsrvector;
  struct Hypre_ProblemDefinition__epv*   e0  = &s_rem__hypre_problemdefinition;
  struct Hypre_SStructBuildVector__epv*  e1  = &s_rem__hypre_sstructbuildvector;
  struct Hypre_Vector__epv*              e2  = &s_rem__hypre_vector;
  struct SIDL_BaseClass__epv*            e3  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv*        e4  = &s_rem__sidl_baseinterface;

  epv->f__cast           = remote_Hypre_SStructParCSRVector__cast;
  epv->f__delete         = remote_Hypre_SStructParCSRVector__delete;
  epv->f__ctor           = NULL;
  epv->f__dtor           = NULL;
  epv->f_addRef          = remote_Hypre_SStructParCSRVector_addRef;
  epv->f_deleteRef       = remote_Hypre_SStructParCSRVector_deleteRef;
  epv->f_isSame          = remote_Hypre_SStructParCSRVector_isSame;
  epv->f_queryInt        = remote_Hypre_SStructParCSRVector_queryInt;
  epv->f_isType          = remote_Hypre_SStructParCSRVector_isType;
  epv->f_getClassInfo    = remote_Hypre_SStructParCSRVector_getClassInfo;
  epv->f_SetCommunicator = remote_Hypre_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize      = remote_Hypre_SStructParCSRVector_Initialize;
  epv->f_Assemble        = remote_Hypre_SStructParCSRVector_Assemble;
  epv->f_GetObject       = remote_Hypre_SStructParCSRVector_GetObject;
  epv->f_SetGrid         = remote_Hypre_SStructParCSRVector_SetGrid;
  epv->f_SetValues       = remote_Hypre_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues    = remote_Hypre_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues     = remote_Hypre_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues  = remote_Hypre_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather          = remote_Hypre_SStructParCSRVector_Gather;
  epv->f_GetValues       = remote_Hypre_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues    = remote_Hypre_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex      = remote_Hypre_SStructParCSRVector_SetComplex;
  epv->f_Print           = remote_Hypre_SStructParCSRVector_Print;
  epv->f_Clear           = remote_Hypre_SStructParCSRVector_Clear;
  epv->f_Copy            = remote_Hypre_SStructParCSRVector_Copy;
  epv->f_Clone           = remote_Hypre_SStructParCSRVector_Clone;
  epv->f_Scale           = remote_Hypre_SStructParCSRVector_Scale;
  epv->f_Dot             = remote_Hypre_SStructParCSRVector_Dot;
  epv->f_Axpy            = remote_Hypre_SStructParCSRVector_Axpy;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e0->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;

  e1->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*)) epv->f__delete;
  e1->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame          = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt        = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType          = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_SetCommunicator = (int32_t (*)(void*,void*)) epv->f_SetCommunicator;
  e1->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e1->f_GetObject       = (int32_t (*)(void*,
    struct SIDL_BaseInterface__object**)) epv->f_GetObject;
  e1->f_SetGrid         = (int32_t (*)(void*,
    struct Hypre_SStructGrid__object*)) epv->f_SetGrid;
  e1->f_SetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_SetValues;
  e1->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_SetBoxValues;
  e1->f_AddToValues     = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,struct SIDL_double__array*)) epv->f_AddToValues;
  e1->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array*)) epv->f_AddToBoxValues;
  e1->f_Gather          = (int32_t (*)(void*)) epv->f_Gather;
  e1->f_GetValues       = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    int32_t,double*)) epv->f_GetValues;
  e1->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct SIDL_int__array*,
    struct SIDL_int__array*,int32_t,
    struct SIDL_double__array**)) epv->f_GetBoxValues;
  e1->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e1->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  e2->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete   = (void (*)(void*)) epv->f__delete;
  e2->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_Clear     = (int32_t (*)(void*)) epv->f_Clear;
  e2->f_Copy      = (int32_t (*)(void*,
    struct Hypre_Vector__object*)) epv->f_Copy;
  e2->f_Clone     = (int32_t (*)(void*,
    struct Hypre_Vector__object**)) epv->f_Clone;
  e2->f_Scale     = (int32_t (*)(void*,double)) epv->f_Scale;
  e2->f_Dot       = (int32_t (*)(void*,struct Hypre_Vector__object*,
    double*)) epv->f_Dot;
  e2->f_Axpy      = (int32_t (*)(void*,double,
    struct Hypre_Vector__object*)) epv->f_Axpy;

  e3->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e3->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e3->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e3->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e3->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e4->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete   = (void (*)(void*)) epv->f__delete;
  e4->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_SStructParCSRVector__object*
Hypre_SStructParCSRVector__remote(const char *url)
{
  struct Hypre_SStructParCSRVector__object* self =
    (struct Hypre_SStructParCSRVector__object*) malloc(
      sizeof(struct Hypre_SStructParCSRVector__object));

  struct Hypre_SStructParCSRVector__object* s0 = self;
  struct SIDL_BaseClass__object*            s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_SStructParCSRVector__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_hypre_problemdefinition.d_epv    = &s_rem__hypre_problemdefinition;
  s0->d_hypre_problemdefinition.d_object = NULL; /* FIXME */

  s0->d_hypre_sstructbuildvector.d_epv    = &s_rem__hypre_sstructbuildvector;
  s0->d_hypre_sstructbuildvector.d_object = NULL; /* FIXME */

  s0->d_hypre_vector.d_epv    = &s_rem__hypre_vector;
  s0->d_hypre_vector.d_object = NULL; /* FIXME */

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_sstructparcsrvector;

  return self;
}
