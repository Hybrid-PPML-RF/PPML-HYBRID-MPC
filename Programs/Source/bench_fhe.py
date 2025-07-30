
import itertools
import random
import math
#import numpy as np
from Compiler import types, library, instructions
from Compiler import comparison, util, ml
from Compiler.util import is_zero, tree_reduce
from Programs.Source import random_matrix
from Compiler import SC_fun

program.set_security(40)

# set the bit length of the cleartext for the comparisons
program.set_bit_length(240)

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

# number of elements in each vector\
    # this captures the different combinations of attributes and attribute values 
    # considered for possible split points.
# n = \alpha * \ceil[\sqrt(m)]    
n = 15              # t = 3
#n = 44             # t = 3
#n = 108            # t = 2
#n = 136            # t = 10
#n = 202            # t = 2
#n = 2048           # t = 7


# n = 8 * \ceil[\sqrt(m)] 


try:
    n = int(program.args[1])
except:
    pass

try:
    t = int(program.args[2])
except:
    pass

try:
    d = int(program.args[3])
except:
    pass

print_ln("Compiled with n=%s, t=%s, d=%s", n, t, d)

if (d==5):
    d5 = 1600
if (d==6):
    d6 = 3100

def compute_gini(a,b,c,d,t):
    #c1 = sint(0)
    #d1 = sint(0)
    #gini = a^2 + b^2 - \sum_{over t} c^2 vals - \sum_{over t} d^2 vals
    @for_range_opt(t)
    def _(j):
        c.square()
        d.square()    
    gini = (a.square() + b.square() - c - d)
    return gini
    
def create_a_b():
    a = types.sint.get_random_int(120, size=1)
    b = types.sint.get_random_int(120, size=1)
    return (a,b)

def create_val():
    val = types.sint.get_random_int(120, size=1)
    val = val.square()
    return  MemValue(val)

def create_a_values(size):
    a = types.sint.get_random_int(120,size=size)
    a = a.square()
    return a

def bench_mul(a,b):
    a*b

def bench_square(a):
    types.sint.square(a)
    
def bench_comp(a,b):
    a < b
    
def bench_argmax(a_values):
    a_max = ml.argmax(a_values)  
    return a_max    


a_array = sint.Array(n)
a_values = create_a_values(size=n)  
a_array.assign(a_values)

"""
# benchmark the argmax 
start_timer(1)
@for_range_opt_multithread(n_threads, d1)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array)
@for_range_opt_multithread(n_threads, d2)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array)
@for_range_opt_multithread(n_threads, d3)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array)
@for_range_opt_multithread(n_threads, d4)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array)  
@for_range_opt_multithread(n_threads, d5)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array) 
@for_range_opt_multithread(n_threads, d6)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(a_array)               
stop_timer(1)
"""

# benchmark computation of GINI index with G' formula of overleaf
#(after FHE preprocessing: n as above )
#(without FHE, TOTAL IN MPC, n as below) 
"""
a = create_val()
b = create_val()
c = create_val()
d = create_val()
start_timer(2)
@for_range_opt_multithread(n_threads, d1*n)
def _(i):
    compute_gini(a,b,c,d,t)     
@for_range_opt_multithread(n_threads, d2*n)
def _(i):
    compute_gini(a,b,c,d,t)
@for_range_opt_multithread(n_threads, d3*n)
def _(i):
    compute_gini(a,b,c,d,t)
@for_range_opt_multithread(n_threads, d4*n)
def _(i):
    compute_gini(a,b,c,d,t)
@for_range_opt_multithread(n_threads, d5*n)
def _(i):
    compute_gini(a,b,c,d,t)
@for_range_opt_multithread(n_threads, d6*n)
def _(i):
    compute_gini(a,b,c,d,t)        
stop_timer(2)
"""

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
x = 205200 # cancer with subset of attributes

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