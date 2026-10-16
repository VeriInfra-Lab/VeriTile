#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
int reached=0;
PyObject *review_parse(PyObject *args){ reached=0;
  // Parse the arguments.
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  int num_warps, num_ctas, shared_memory;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  PyObject *arg_annotations = NULL;
  Py_buffer signature;
  PyObject *kernel_args = NULL;
  if (!PyArg_ParseTuple(args, "iiiKKpp(iii)OOOOOOy*O", &gridX, &gridY, &gridZ,
                        &_stream, &_function, &launch_cooperative_grid,
                        &launch_pdl, &num_warps, &num_ctas, &shared_memory,
                        &launch_metadata, &launch_enter_hook, &launch_exit_hook,
                        &global_scratch_obj, &profile_scratch_obj,
                        &arg_annotations, &signature, &kernel_args)) {
    return NULL;
  }


reached=1;PyBuffer_Release(&signature);PyErr_SetString(PyExc_ValueError,"review downstream marker");return NULL;}
bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}
uint64_t review_ptr(uint64_t data,uint32_t width,uint32_t block,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)(pid*block+lane)*width;}
