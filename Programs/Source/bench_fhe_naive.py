
import itertools
import random
import math
#import numpy as np
from Compiler import types, library, instructions
from Compiler import comparison, util, ml
from Compiler.util import is_zero, tree_reduce
from Compiler import SC_fun

program.set_security(40)

# set the bit length of the cleartext for the comparisons
program.set_bit_length(64)

# Set the bit length based on edabit
bit_length = program.bit_length
print_ln("%s-bit_length", bit_length)


# depth of the tree
d=4
# argmax to be computed collaboratively between the servers
# in parallel, on each tree level 
# e.g. if we assume 10 servers, each one training 10 decision trees\
    # then, on the first level d1 = 100 argmaxes can be computed independently, 
    # and on the second level d2=200 argmaxes in parallel, etc.. 
d1 = 100
d2 = 200 
d3 = 400
d4 = 800
d5 = 0
d6 = 0


n_threads = 48

# number of labels, needed for the computation of gini
#t = 2
t = 3
#t = 7
#t = 10

x=12600

try:
    x = int(program.args[1])
except:
    pass

try:
    d = int(program.args[2])
except:
    pass


print_ln("Compiled with x=%s, d=%s", x, d)
if (d==5):
    d5 = 1600
if (d==6):
    d6 = 3100


def create_val():
    val = types.sint.get_random_int(120, size=1)
    val = val.square()
    return  MemValue(val)


def bench_mul(a,b):
    a*b
    
# benchmark computation of GINI index with G' formula of overleaf
#(without FHE, TOTAL IN MPC, n as below) 
#"""
a = create_val()
b = create_val()
#x = 25200  # iris, 
#x = 12600 # iris with subset of attributes
#x = 119119 # wine
#x = 36652 # wine with subset of attributes
#x = 1026000 # cancer
#x = 205200 # cancer with subset of attributes

start_timer(3)
@for_range_opt_multithread(n_threads, d1*x)
def _(i):
    bench_mul(a,b)
@for_range_opt_multithread(n_threads, d2*x)
def _(i):
    bench_mul(a,b)
@for_range_opt_multithread(n_threads, d3*x)
def _(i):
    bench_mul(a,b)
@for_range_opt_multithread(n_threads, d4*x)
def _(i):
    bench_mul(a,b)
@for_range_opt_multithread(n_threads, d5*x)
def _(i):
    bench_mul(a,b)
@for_range_opt_multithread(n_threads, d6*x)
def _(i):
    bench_mul(a,b)      
stop_timer(3)
#"""
