/*
 * File:          Hypre_SStructGrid_IOR.c
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:23 PST
 * Generated:     20030306 17:05:25 PST
 * Description:   Intermediate Object Representation for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "Hypre_SStructGrid_IOR.h"
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

static struct Hypre_SStructGrid__epv s_new__hypre_sstructgrid;
static struct Hypre_SStructGrid__epv s_rem__hypre_sstructgrid;

static struct SIDL_BaseClass__epv  s_new__sidl_baseclass;
static struct SIDL_BaseClass__epv* s_old__sidl_baseclass;
static struct SIDL_BaseClass__epv  s_rem__sidl_baseclass;

static struct SIDL_BaseInterface__epv  s_new__sidl_baseinterface;
static struct SIDL_BaseInterface__epv* s_old__sidl_baseinterface;
static struct SIDL_BaseInterface__epv  s_rem__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

extern void Hypre_SStructGrid__set_epv(
  struct Hypre_SStructGrid__epv* epv);

/*
 * CAST: dynamic type casting support.
 */

static void* ior_Hypre_SStructGrid__cast(
  struct Hypre_SStructGrid__object* self,
  const char* name)
{
  void* cast = NULL;

  struct Hypre_SStructGrid__object* s0 = self;
  struct SIDL_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "Hypre.SStructGrid")) {
    cast = (void*) s0;
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

static void ior_Hypre_SStructGrid__delete(
  struct Hypre_SStructGrid__object* self)
{
  Hypre_SStructGrid__fini(self);
  memset((void*)self, 0, sizeof(struct Hypre_SStructGrid__object));
  free((void*) self);
}

/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void Hypre_SStructGrid__init_epv(
  struct Hypre_SStructGrid__object* self)
{
  struct Hypre_SStructGrid__object* s0 = self;
  struct SIDL_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  struct Hypre_SStructGrid__epv*  epv = &s_new__hypre_sstructgrid;
  struct SIDL_BaseClass__epv*     e0  = &s_new__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_new__sidl_baseinterface;

  s_old__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old__sidl_baseclass     = s1->d_epv;

  epv->f__cast               = ior_Hypre_SStructGrid__cast;
  epv->f__delete             = ior_Hypre_SStructGrid__delete;
  epv->f__ctor               = NULL;
  epv->f__dtor               = NULL;
  epv->f_addRef              = (void (*)(struct Hypre_SStructGrid__object*)) 
    s1->d_epv->f_addRef;
  epv->f_deleteRef           = (void (*)(struct Hypre_SStructGrid__object*)) 
    s1->d_epv->f_deleteRef;
  epv->f_isSame              = (SIDL_bool (*)(struct Hypre_SStructGrid__object*,
    struct SIDL_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt            = (struct SIDL_BaseInterface__object* (*)(struct 
    Hypre_SStructGrid__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType              = (SIDL_bool (*)(struct Hypre_SStructGrid__object*,
    const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo        = (struct SIDL_ClassInfo__object* (*)(struct 
    Hypre_SStructGrid__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetNumDimParts      = NULL;
  epv->f_SetExtents          = NULL;
  epv->f_SetVariable         = NULL;
  epv->f_AddVariable         = NULL;
  epv->f_SetNeighborBox      = NULL;
  epv->f_AddUnstructuredPart = NULL;
  epv->f_SetPeriodic         = NULL;
  epv->f_SetNumGhost         = NULL;

  Hypre_SStructGrid__set_epv(epv);

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete   = (void (*)(void*)) epv->f__delete;
  e1->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

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
      SIDL_ClassInfoI_setName(impl, "Hypre.SStructGrid");
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
initMetadata(struct Hypre_SStructGrid__object* self)
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

struct Hypre_SStructGrid__object*
Hypre_SStructGrid__new(void)
{
  struct Hypre_SStructGrid__object* self =
    (struct Hypre_SStructGrid__object*) malloc(
      sizeof(struct Hypre_SStructGrid__object));
  Hypre_SStructGrid__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void Hypre_SStructGrid__init(
  struct Hypre_SStructGrid__object* self)
{
  struct Hypre_SStructGrid__object* s0 = self;
  struct SIDL_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  SIDL_BaseClass__init(s1);

  if (!s_method_initialized) {
    Hypre_SStructGrid__init_epv(s0);
  }

  s1->d_sidl_baseinterface.d_epv = &s_new__sidl_baseinterface;
  s1->d_epv                      = &s_new__sidl_baseclass;

  s0->d_epv    = &s_new__hypre_sstructgrid;

  s0->d_data = NULL;

  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void Hypre_SStructGrid__fini(
  struct Hypre_SStructGrid__object* self)
{
  struct Hypre_SStructGrid__object* s0 = self;
  struct SIDL_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old__sidl_baseinterface;
  s1->d_epv                      = s_old__sidl_baseclass;

  SIDL_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
Hypre_SStructGrid__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}
static const struct Hypre_SStructGrid__external
s_externalEntryPoints = {
  Hypre_SStructGrid__new,
  Hypre_SStructGrid__remote,
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_SStructGrid__external*
Hypre_SStructGrid__externals(void)
{
  return &s_externalEntryPoints;
}

/*
 * REMOTE CAST: dynamic type casting for remote objects.
 */

static void* remote_Hypre_SStructGrid__cast(
  struct Hypre_SStructGrid__object* self,
  const char* name)
{
  return NULL;
}

/*
 * REMOTE DELETE: call the remote destructor for the object.
 */

static void remote_Hypre_SStructGrid__delete(
  struct Hypre_SStructGrid__object* self)
{
  free((void*) self);
}

/*
 * REMOTE METHOD STUB:addRef
 */

static void
remote_Hypre_SStructGrid_addRef(
  struct Hypre_SStructGrid__object* self)
{
}

/*
 * REMOTE METHOD STUB:deleteRef
 */

static void
remote_Hypre_SStructGrid_deleteRef(
  struct Hypre_SStructGrid__object* self)
{
}

/*
 * REMOTE METHOD STUB:isSame
 */

static SIDL_bool
remote_Hypre_SStructGrid_isSame(
  struct Hypre_SStructGrid__object* self,
  struct SIDL_BaseInterface__object* iobj)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:queryInt
 */

static struct SIDL_BaseInterface__object*
remote_Hypre_SStructGrid_queryInt(
  struct Hypre_SStructGrid__object* self,
  const char* name)
{
  return (struct SIDL_BaseInterface__object*) 0;
}

/*
 * REMOTE METHOD STUB:isType
 */

static SIDL_bool
remote_Hypre_SStructGrid_isType(
  struct Hypre_SStructGrid__object* self,
  const char* name)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:getClassInfo
 */

static struct SIDL_ClassInfo__object*
remote_Hypre_SStructGrid_getClassInfo(
  struct Hypre_SStructGrid__object* self)
{
  return (struct SIDL_ClassInfo__object*) 0;
}

/*
 * REMOTE METHOD STUB:SetNumDimParts
 */

static int32_t
remote_Hypre_SStructGrid_SetNumDimParts(
  struct Hypre_SStructGrid__object* self,
  int32_t ndim,
  int32_t nparts)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetExtents
 */

static int32_t
remote_Hypre_SStructGrid_SetExtents(
  struct Hypre_SStructGrid__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetVariable
 */

static int32_t
remote_Hypre_SStructGrid_SetVariable(
  struct Hypre_SStructGrid__object* self,
  int32_t part,
  int32_t var,
  enum Hypre_SStructVariable__enum vartype)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddVariable
 */

static int32_t
remote_Hypre_SStructGrid_AddVariable(
  struct Hypre_SStructGrid__object* self,
  int32_t part,
  struct SIDL_int__array* index,
  int32_t var,
  enum Hypre_SStructVariable__enum vartype)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNeighborBox
 */

static int32_t
remote_Hypre_SStructGrid_SetNeighborBox(
  struct Hypre_SStructGrid__object* self,
  int32_t part,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t nbor_part,
  struct SIDL_int__array* nbor_ilower,
  struct SIDL_int__array* nbor_iupper,
  struct SIDL_int__array* index_map)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:AddUnstructuredPart
 */

static int32_t
remote_Hypre_SStructGrid_AddUnstructuredPart(
  struct Hypre_SStructGrid__object* self,
  int32_t ilower,
  int32_t iupper)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetPeriodic
 */

static int32_t
remote_Hypre_SStructGrid_SetPeriodic(
  struct Hypre_SStructGrid__object* self,
  int32_t part,
  struct SIDL_int__array* periodic)
{
  return 0;
}

/*
 * REMOTE METHOD STUB:SetNumGhost
 */

static int32_t
remote_Hypre_SStructGrid_SetNumGhost(
  struct Hypre_SStructGrid__object* self,
  struct SIDL_int__array* num_ghost)
{
  return 0;
}

/*
 * REMOTE EPV: create remote entry point vectors (EPVs).
 */

static void Hypre_SStructGrid__init_remote_epv(void)
{
  struct Hypre_SStructGrid__epv*  epv = &s_rem__hypre_sstructgrid;
  struct SIDL_BaseClass__epv*     e0  = &s_rem__sidl_baseclass;
  struct SIDL_BaseInterface__epv* e1  = &s_rem__sidl_baseinterface;

  epv->f__cast               = remote_Hypre_SStructGrid__cast;
  epv->f__delete             = remote_Hypre_SStructGrid__delete;
  epv->f__ctor               = NULL;
  epv->f__dtor               = NULL;
  epv->f_addRef              = remote_Hypre_SStructGrid_addRef;
  epv->f_deleteRef           = remote_Hypre_SStructGrid_deleteRef;
  epv->f_isSame              = remote_Hypre_SStructGrid_isSame;
  epv->f_queryInt            = remote_Hypre_SStructGrid_queryInt;
  epv->f_isType              = remote_Hypre_SStructGrid_isType;
  epv->f_getClassInfo        = remote_Hypre_SStructGrid_getClassInfo;
  epv->f_SetNumDimParts      = remote_Hypre_SStructGrid_SetNumDimParts;
  epv->f_SetExtents          = remote_Hypre_SStructGrid_SetExtents;
  epv->f_SetVariable         = remote_Hypre_SStructGrid_SetVariable;
  epv->f_AddVariable         = remote_Hypre_SStructGrid_AddVariable;
  epv->f_SetNeighborBox      = remote_Hypre_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart = remote_Hypre_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic         = remote_Hypre_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost         = remote_Hypre_SStructGrid_SetNumGhost;

  e0->f__cast        = (void* (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f__delete;
  e0->f_addRef       = (void (*)(struct SIDL_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct SIDL_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct SIDL_BaseInterface__object* (*)(struct 
    SIDL_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (SIDL_bool (*)(struct SIDL_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct SIDL_ClassInfo__object* (*)(struct 
    SIDL_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast     = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete   = (void (*)(void*)) epv->f__delete;
  e1->f_addRef    = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame    = (SIDL_bool (*)(void*,
    struct SIDL_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt  = (struct SIDL_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType    = (SIDL_bool (*)(void*,const char*)) epv->f_isType;

  s_remote_initialized = 1;
}

/*
 * REMOTE: generate remote instance given URL string.
 */

struct Hypre_SStructGrid__object*
Hypre_SStructGrid__remote(const char *url)
{
  struct Hypre_SStructGrid__object* self =
    (struct Hypre_SStructGrid__object*) malloc(
      sizeof(struct Hypre_SStructGrid__object));

  struct Hypre_SStructGrid__object* s0 = self;
  struct SIDL_BaseClass__object*    s1 = &s0->d_sidl_baseclass;

  if (!s_remote_initialized) {
    Hypre_SStructGrid__init_remote_epv();
  }

  s1->d_sidl_baseinterface.d_epv    = &s_rem__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = NULL; /* FIXME */

  s1->d_data = NULL; /* FIXME */
  s1->d_epv  = &s_rem__sidl_baseclass;

  s0->d_data = NULL; /* FIXME */
  s0->d_epv  = &s_rem__hypre_sstructgrid;

  return self;
}
