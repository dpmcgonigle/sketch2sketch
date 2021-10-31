#! /usr/bin/env python

from expand_data import debug_functions, print_debug
import inspect

############################################################################################                                                 
def get_options():                                                                                                                           
    """                                                                                                                                      
    return command line options as an argparse.parse_args() object                                                                           
    if called from jupyter,                                                                                                                  
    """                                                                                                                                      
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str, default=None, help="Comma-separated list of functions (no spaces); e.g. func1,func2")
    # If you want to call this get_args() from a Jupyter Notebook, you need to uncomment -f line. Them's the rules.                          
    #parser.add_argument('-f', '--file', help='Path for input file.')                                                                        
    return parser.parse_args()                             

def func():
    print_debug("Inside func()")

def wrapfunc():
    print_debug("Inside wrapfunc()")
    func()

if __name__=="__main__":
    print_debug("inside __main__")
    wrapfunc()
