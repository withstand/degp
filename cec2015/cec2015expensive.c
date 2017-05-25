//#include "Python.h"
#ifdef _DEBUG
  #undef _DEBUG
  #include <Python.h>
  #define _DEBUG
#else
  #include <Python.h>
#endif
#include "src/cec15_test_func.h"

//static PyObject *
//cec2015_system(PyObject *self, PyObject *args)
//{
//    const char *command;
//    int sts;
//
//    if (!PyArg_ParseTuple(args, "s", &command))
//        return NULL;
//    sts = system(command);
//    return PyLong_FromLong(sts);
//}

//void cec15_test_func(double *x, double *f, int nx, int mx,int func_num);

static PyObject *
cec2015_func(PyObject *self, PyObject *args)
{
    double * x;
    int n;

    double ret;
    int m = 1;
    int func_num;
    
    PyObject *xobj = PySequence_GetItem(args,0);
    PyObject *f_num = PySequence_GetItem(args,1);
    
    if (!PySequence_Check(xobj)){return NULL;}
    Py_ssize_t size = PySequence_Size(xobj);
                                     
    n = (int)size;
    x = (double*) malloc(sizeof( double) *n);
    //printf("%d :", n);
    for (Py_ssize_t i=0; i<n; i++){
        x[i] = PyFloat_AsDouble(PySequence_GetItem(xobj, i));
        //printf("%f ", x[i]);
    }
    //printf("\n");
         
    func_num = PyLong_AsLong(f_num);                   
    
    cec15_test_func(x, &ret, n, m, func_num);
    free(x);
    return PyFloat_FromDouble(ret);
}


static PyMethodDef cec2015_methods[] = {
//     {"system",  cec2015_system, METH_VARARGS,
//	     "Execute a shell command."},
     {"func",  cec2015_func, METH_VARARGS,
	     "Evaluate a test function."},
     {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cec2015module = {
   PyModuleDef_HEAD_INIT,
   "cec2015",   /* name of module */
   "cec2015 expensive contest test functions", /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   cec2015_methods
};



PyMODINIT_FUNC PyInit_cec2015expensive(void) 

{	
	PyObject *m;
	m = PyModule_Create(&cec2015module);
	return m;
}
