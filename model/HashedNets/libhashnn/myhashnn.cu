
// #include "luaT.h"
// #include <lua.hpp>
// #include "lua.h"
// #include "lualib.h"
// #include "lauxlib.h"


#ifdef __cplusplus
extern "C" {
#endif
/*#include "utils.h"*/
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#include "luaT.h"
#include "TH.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */
#include "THCGeneral.h"
#include <assert.h>
#ifdef __cplusplus

}
#endif

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/find.h>
#include <iostream>


THCState* getCutorchState(lua_State* L)
{
    lua_getglobal(L, "cutorch");
    lua_getfield(L, -1, "getState");
    lua_call(L, 0, 1);
    THCState *state = (THCState*) lua_touserdata(L, -1);
    lua_pop(L, 2);
    return state;
}

struct myindex_functor
{
  /*const thrust:device_ptr<float> Whash_ptr;*/
  const float* Whash_ptr;

  myindex_functor(float* _ptr) : Whash_ptr(_ptr) {}

  __host__ __device__
    float operator()(const float& ind) const { 
      return *(Whash_ptr + (long)ind - 1);
      /*return (long)ind;*/
    }
};


/* First Input is Whash vector. Second input is Wind*/
int libhashnn_myindexing(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *myWhash = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *myWind = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *myWmatrix = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  thrust::device_ptr<float> myWhash_data(THCudaTensor_data(state, myWhash));
  thrust::device_ptr<float> myWind_data(THCudaTensor_data(state, myWind));
  thrust::device_ptr<float> myW_data(THCudaTensor_data(state, myWmatrix));

  long fullsize = THCudaTensor_nElement(state, myWmatrix);
  long hashsize = THCudaTensor_nElement(state, myWhash);
  myWhash = THCudaTensor_newContiguous(state, myWhash);
  myWmatrix = THCudaTensor_newContiguous(state, myWmatrix);
  myWind = THCudaTensor_newContiguous(state, myWind);

  thrust::transform(myWind_data, myWind_data+fullsize, myW_data, myindex_functor(thrust::raw_pointer_cast(myWhash_data)));
  /*thrust::transform(myWind_data, myWind_data+fullsize, myW_data, myindex_functor());*/

  return 0;
}



int libhashnn_mysort(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *key = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *val = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  thrust::device_ptr<float> key_data(THCudaTensor_data(state, key));
  thrust::device_ptr<float> val_data(THCudaTensor_data(state, val));

  long size = THCudaTensor_nElement(state, key);
  key = THCudaTensor_newContiguous(state, key);
  val = THCudaTensor_newContiguous(state, val);

  /*Sort by Key*/
  thrust::sort_by_key(key_data, key_data+size, val_data);

  return 0;
}

struct rearrange_functor
{
  /*const thrust:device_ptr<float> Whash_ptr;*/
  float* hash_grad_ptr;

  rearrange_functor(float* _ptr) : hash_grad_ptr(_ptr) {}

  __host__ __device__
    float operator()(const float& unique_key, const float& buffer) const{
      if (unique_key!=0) {
        *(hash_grad_ptr + (long)unique_key - 1) = buffer;
      }
      return buffer;
    }
};


int libhashnn_myreduce(lua_State *L) {
  THCState *state = getCutorchState(L);

  THCudaTensor *sorted_key = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *full_grad = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *unique_key = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *hash_grad = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *buffer = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
  /*bool average_flag = false;*/
  /*if (lua_isboolean(L, 5)) {*/
  /*  average_flag = (bool)lua_toboolean(L,5);*/
  /*}*/

  thrust::device_ptr<float> sorted_key_data(THCudaTensor_data(state, sorted_key));
  thrust::device_ptr<float> full_grad_data(THCudaTensor_data(state, full_grad));
  thrust::device_ptr<float> unique_key_data(THCudaTensor_data(state, unique_key));
  thrust::device_ptr<float> hash_grad_data(THCudaTensor_data(state, hash_grad));
  thrust::device_ptr<float> buffer_data(THCudaTensor_data(state, buffer));

  long size = THCudaTensor_nElement(state, full_grad);
  long hashsize = THCudaTensor_nElement(state, hash_grad);
  sorted_key = THCudaTensor_newContiguous(state, sorted_key);
  full_grad = THCudaTensor_newContiguous(state, full_grad);
  unique_key = THCudaTensor_newContiguous(state, unique_key);
  hash_grad = THCudaTensor_newContiguous(state, hash_grad);
  buffer = THCudaTensor_newContiguous(state, buffer);

  /*Reduce by Key*/
  /*thrust::equal_to<float> binary_pred;*/
  thrust::reduce_by_key(sorted_key_data, sorted_key_data+size, full_grad_data
                        ,unique_key_data,buffer_data);

  thrust::transform(unique_key_data, unique_key_data+hashsize, buffer_data, buffer_data, rearrange_functor(thrust::raw_pointer_cast(hash_grad_data)));

  return 0;
}


/*static const struct luaL_reg libhashnnlib[] = {*/
static const struct luaL_Reg libhashnnlib[] = {
  {"myindexing", libhashnn_myindexing},
  {"myreduce", libhashnn_myreduce},
  {"mysort", libhashnn_mysort},
  {NULL, NULL}
};


extern "C"
int luaopen_libhashnn(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, libhashnnlib, "libhashnn");
  lua_pop(L,1);
  luaL_register(L,"libhashnn",libhashnnlib);
  return 1;
}




