#include <Python.h>
#include <stdint.h>
int pointer_attempted=0;
PyObject *parse_then_failure(PyObject *args) {
  pointer_attempted=0;
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

  pointer_attempted=1;
  PyBuffer_Release(&signature);
  PyErr_SetString(PyExc_ValueError,"instrumented downstream pointer-packing failure");
  return NULL;
}
