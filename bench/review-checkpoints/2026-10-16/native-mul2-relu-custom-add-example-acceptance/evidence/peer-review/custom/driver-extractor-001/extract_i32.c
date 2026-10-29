#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

