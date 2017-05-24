from distutils.core import setup, Extension

module1 = Extension('cec2015expensive',
                    sources = ['cec2015expensive.c','src/cec15_test_func.c'])

setup (name = 'cec2015expensive',
       version = '1.0',
       description = 'This is a package of cec2015 single objective competetion expensive functions',
       ext_modules = [module1])
