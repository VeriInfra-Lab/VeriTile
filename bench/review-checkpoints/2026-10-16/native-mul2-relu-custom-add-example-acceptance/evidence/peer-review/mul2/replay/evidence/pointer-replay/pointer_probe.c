#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
typedef uint64_t CUdeviceptr;
typedef int CUresult;
#define CUDA_ERROR_INVALID_VALUE 1
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 3
static CUdeviceptr requested_mapping;
static int cuda_calls;
static CUresult cuPointerGetAttribute(CUdeviceptr *out, int attr, CUdeviceptr input) {
 cuda_calls++; *out=requested_mapping; return 0;
}
static bool gpuAssert(CUresult s, const char *file, int line) {return s==0;}
static PyObject *data_ptr_str = NULL;

// Extract a CUDA device pointer from a pointer-like PyObject obj, and store
// it to the memory location pointed by ptr.
bool extractPointer(void *ptr, PyObject *obj) {
  CUdeviceptr *dev_ptr = ptr;
  if (obj == Py_None) {
    *dev_ptr = (CUdeviceptr)0; // valid nullptr
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return true;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(
        PyExc_TypeError,
        "Pointer argument must be either uint64 or have data_ptr method");
    return false;
  }
  if (!PyLong_Check(ret)) {
    PyErr_SetString(PyExc_TypeError,
                    "data_ptr method of Pointer object must return 64-bit int");
    return false;
  }
  *dev_ptr = PyLong_AsUnsignedLongLong(ret);
  Py_DECREF(ret);
  if (*dev_ptr == 0) {
    return true; // valid nullptr
  }
  CUresult status = cuPointerGetAttribute(
      dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
    PyErr_Format(PyExc_ValueError,
                 "Pointer argument cannot be accessed from Triton "
                 "(cpu tensor?)");
    return false;
  }
  return gpuAssert(status, __FILE__, __LINE__);
}

bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

static PyObject *probe(PyObject *self, PyObject *args) {
 PyObject *obj; unsigned long long mapped;
 if (!PyArg_ParseTuple(args,"OK", &obj,&mapped)) return NULL;
 if (!data_ptr_str) data_ptr_str=PyUnicode_InternFromString("data_ptr");
 requested_mapping=mapped;cuda_calls=0;CUdeviceptr out=0;
 bool ok=extractPointer(&out,obj);
 if (!ok || PyErr_Occurred()) return NULL;
 return Py_BuildValue("Ki",out,cuda_calls);
}
static PyObject *grid(PyObject *self, PyObject *args) {
 int n; if(!PyArg_ParseTuple(args,"i",&n)) return NULL;return PyLong_FromLong(n);
}
static PyObject *narrow(PyObject *self, PyObject *obj) {
 int32_t n;if(!extractI32(&n,obj))return NULL;return PyLong_FromLong(n);
}
static PyMethodDef methods[]={{"pointer",probe,METH_VARARGS,""},{"grid",grid,METH_VARARGS,""},{"narrow",narrow,METH_O,""},{NULL,NULL,0,NULL}};
static struct PyModuleDef mod={PyModuleDef_HEAD_INIT,"pointer_probe",NULL,-1,methods};
PyMODINIT_FUNC PyInit_pointer_probe(void){return PyModule_Create(&mod);}
