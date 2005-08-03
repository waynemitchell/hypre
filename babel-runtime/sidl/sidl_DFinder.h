/*
 * File:          sidl_DFinder.h
 * Symbol:        sidl.DFinder-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side glue code for sidl.DFinder
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_sidl_DFinder_h
#define included_sidl_DFinder_h

/**
 * Symbol "sidl.DFinder" (version 0.9.3)
 * 
 *  This class is the Default Finder.  If no Finder is set in class Loader,
 *  this finder is used.  It uses SCL files from the filesystem to
 *  resolve dynamic libraries.
 * 
 * The initial search path is taken from the SIDL_DLL_PATH
 * environment variable.
 */
struct sidl_DFinder__object;
struct sidl_DFinder__array;
typedef struct sidl_DFinder__object* sidl_DFinder;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_DLL_h
#include "sidl_DLL.h"
#endif
#ifndef included_sidl_Resolve_h
#include "sidl_Resolve.h"
#endif
#ifndef included_sidl_Scope_h
#include "sidl_Scope.h"
#endif

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct sidl_DFinder__object*
sidl_DFinder__create(void);

/**
 * RMI constructor function for the class.
 */
sidl_DFinder
sidl_DFinder__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
sidl_DFinder
sidl_DFinder__connect(const char *, sidl_BaseInterface *_ex);
/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
void
sidl_DFinder_addRef(
  /* in */ sidl_DFinder self);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
sidl_DFinder_deleteRef(
  /* in */ sidl_DFinder self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
sidl_bool
sidl_DFinder_isSame(
  /* in */ sidl_DFinder self,
  /* in */ sidl_BaseInterface iobj);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
sidl_BaseInterface
sidl_DFinder_queryInt(
  /* in */ sidl_DFinder self,
  /* in */ const char* name);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
sidl_bool
sidl_DFinder_isType(
  /* in */ sidl_DFinder self,
  /* in */ const char* name);

/**
 * Return the meta-data about the class implementing this interface.
 */
sidl_ClassInfo
sidl_DFinder_getClassInfo(
  /* in */ sidl_DFinder self);

/**
 * Find a DLL containing the specified information for a sidl
 * class. This method searches through the files in set set path
 * looking for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 *                   class/interface to be found. Package names
 *                   are separated by period characters from each
 *                   other and the class/interface name.
 * @param target     to find a client-side binding, this is
 *                   normally the name of the language.
 *                   To find the implementation of a class
 *                   in order to make one, you should pass
 *                   the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 *                   be loaded into the global scope, a local
 *                   scope, or use the setting in the file.
 * @param lResolve   this specifies whether symbols should be
 *                   resolved as needed (LAZY), completely
 *                   resolved at load time (NOW), or use the
 *                   setting from the file.
 * @return a non-NULL object means the search was successful.
 *         The DLL has already been added.
 */
sidl_DLL
sidl_DFinder_findLibrary(
  /* in */ sidl_DFinder self,
  /* in */ const char* sidl_name,
  /* in */ const char* target,
  /* in */ enum sidl_Scope__enum lScope,
  /* in */ enum sidl_Resolve__enum lResolve);

/**
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 */
void
sidl_DFinder_setSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_name);

/**
 * Return the current search path.  If the search path has not been
 * set, then the search path will be taken from environment variable
 * SIDL_DLL_PATH.
 */
char*
sidl_DFinder_getSearchPath(
  /* in */ sidl_DFinder self);

/**
 * Append the specified path fragment to the beginning of the
 * current search path.  If the search path has not yet been set
 * by a call to <code>setSearchPath</code>, then this fragment will
 * be appended to the path in environment variable SIDL_DLL_PATH.
 */
void
sidl_DFinder_addSearchPath(
  /* in */ sidl_DFinder self,
  /* in */ const char* path_fragment);

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_DFinder__object*
sidl_DFinder__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
sidl_DFinder__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
sidl_DFinder__exec(
  /* in */ sidl_DFinder self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
sidl_DFinder__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
sidl_DFinder__getURL(
  /* in */ sidl_DFinder self);
/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_create1d(int32_t len);

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_create1dInit(
  int32_t len, 
  sidl_DFinder* data);

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_borrow(
  sidl_DFinder* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_smartCopy(
  struct sidl_DFinder__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
sidl_DFinder__array_addRef(
  struct sidl_DFinder__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidl_DFinder__array_deleteRef(
  struct sidl_DFinder__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get1(
  const struct sidl_DFinder__array* array,
  const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get2(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get3(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get4(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get5(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get6(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidl_DFinder
sidl_DFinder__array_get7(
  const struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidl_DFinder
sidl_DFinder__array_get(
  const struct sidl_DFinder__array* array,
  const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidl_DFinder__array_set1(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  sidl_DFinder const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidl_DFinder__array_set2(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  sidl_DFinder const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidl_DFinder__array_set3(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidl_DFinder const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidl_DFinder__array_set4(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidl_DFinder const value);

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidl_DFinder__array_set5(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidl_DFinder const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidl_DFinder__array_set6(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidl_DFinder const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidl_DFinder__array_set7(
  struct sidl_DFinder__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidl_DFinder const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidl_DFinder__array_set(
  struct sidl_DFinder__array* array,
  const int32_t indices[],
  sidl_DFinder const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidl_DFinder__array_dimen(
  const struct sidl_DFinder__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DFinder__array_lower(
  const struct sidl_DFinder__array* array,
  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DFinder__array_upper(
  const struct sidl_DFinder__array* array,
  const int32_t ind);

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DFinder__array_length(
  const struct sidl_DFinder__array* array,
  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidl_DFinder__array_stride(
  const struct sidl_DFinder__array* array,
  const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_DFinder__array_isColumnOrder(
  const struct sidl_DFinder__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidl_DFinder__array_isRowOrder(
  const struct sidl_DFinder__array* array);

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_slice(
  struct sidl_DFinder__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
sidl_DFinder__array_copy(
  const struct sidl_DFinder__array* src,
  struct sidl_DFinder__array* dest);

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct sidl_DFinder__array*
sidl_DFinder__array_ensure(
  struct sidl_DFinder__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif