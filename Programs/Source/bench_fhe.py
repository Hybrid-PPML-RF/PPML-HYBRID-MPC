# number of labels, needed for the computation of gini
# t = 2
# t = 3
# t = 7
# t = 10

# number of elements in each vector\
    # this captures the different combinations of attributes and attribute values 
    # considered for possible split points.
# ma = \alpha * \ceil[\sqrt(m)]    
# ma = 15             # t = 3
# ma = 44             # t = 3
# ma = 108            # t = 2
# ma = 136            # t = 10
# ma = 202            # t = 2
# ma = 2048           # t = 7
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
program.set_bit_length(32)

# Set the bit length based on edabit
bit_length = program.bit_length
print_ln("%s-bit_length", bit_length)

#default parameters for the benchmarks, which can be overwritten at compile time by passing arguments to the compiler
n = 1 # number of samples
m = 9   # number of attributes, needed for the computation of the number of possible split points
alpha = 4 # number of possible attribute values, needed for the computation of the number of possible split points
t = 3   # number of labels
d = 1   # depth of the tree

# argmax to be computed collaboratively between the servers
# in parallel, on each tree level 
# e.g. if we assume 10 servers, each one training 10 decision trees\
    # then, on the first level d1 = 100 argmaxes can be computed independently, 
    # and on the second level d2=200 argmaxes in parallel, etc.. 
d1 = 100
d2 = 0 
d3 = 0
d4 = 0
d5 = 0
d6 = 0
n_threads = 4

# n = 8 * \ceil[\sqrt(m)] 

try:
    n = int(program.args[1])
except:
    pass

try:
    m = int(program.args[2])
except:
    pass

try:
    alpha = int(program.args[3])
except:
    pass

try:
    t = int(program.args[4])
except:
    pass

try:
    d = int(program.args[5])
except:
    pass

ma = alpha * math.ceil(math.sqrt(m)) # number of elements in each vector, which captures the different combinations of attributes and attribute values considered for possible split points. This is needed for the computation of the gini index, as well as for the argmax.

print_ln("Compiled with n=%s, m=%s, alpha=%s, t=%s, d=%s", n, m, alpha, t, d)


if (d==5):
    d5 = 1600
if (d==6):
    d5 = 1600
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
    a = types.sint.get_random_int(32)
    b = types.sint.get_random_int(32)
    return (MemValue(a),MemValue(b))

def create_val():
    val = types.sint.get_random_int(32)
    val = val.square()
    return  MemValue(val)

def create_a_values(n):
    a = types.sint.get_random_int(32,size=n)
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
    return a_max.reveal()    

def bench_max(a_values):
    a_max = ml.argmax(a_values)  
    idx = a_max.reveal()   
    return a_values[idx].reveal()


a_array = sint.Array(ma)
a_values = create_a_values(ma)  
a_array.assign(a_values)

t_array = sint.Array(t)
t_values = create_a_values(t)  
t_array.assign(t_values)

# benchmark the argmax 
start_timer(1)

@for_range_opt_multithread(n_threads, d1)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = ml.argmax(a_array)
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
    if (d5==0):
        a_max = bench_argmax(t_array)
    if (d5!=0):
        a_max = bench_argmax(a_array)      
@for_range_opt_multithread(n_threads, d5)
def _(i):
    #@for_range(l)
    #def _(i):
    if (d6==0):
        a_max = bench_argmax(t_array)
    if (d6!=0):
        a_max = bench_argmax(a_array)
@for_range_opt_multithread(n_threads, d6)
def _(i):
    #@for_range(l)
    #def _(i):
    a_max = bench_argmax(t_array)               
stop_timer(1)

# benchmark computation of GINI index with G' formula of overleaf
#(after FHE preprocessing: n as above )
#(without FHE, TOTAL IN MPC, n as below) 
aval = create_val()
bval = create_val()
cval = create_val()
dval = create_val()
start_timer(2)
@for_range_opt_multithread(n_threads, d1*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)     
@for_range_opt_multithread(n_threads, d2*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)
@for_range_opt_multithread(n_threads, d3*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)
@for_range_opt_multithread(n_threads, d4*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)
@for_range_opt_multithread(n_threads, d5*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)
@for_range_opt_multithread(n_threads, d6*ma)
def _(i):
    compute_gini(aval,bval,cval,dval,t)        
stop_timer(2)

#benchmark for input integrity
(x,y) = create_a_b()
tt = d*n*m*alpha + d*(n*m + t)*(alpha + 1) + d*n*m*t # number of multiplications in the whole tree, for the given parameters, as a function of n, t, d, and samples. This is used for benchmarking the input integrity check.
#input integrity
start_timer(4)
@for_range_opt_multithread(n_threads, tt)
def _(i):
    z = x*y
stop_timer(4)


# benchmark input sharing
# S = 2*(T*alpha*n*m) + 4*T*(2^d-1)*alpha*floor(sqrt(m))*t*n
#       + T*(2^d-1)*n*t + T*(2^d-1)*alpha*floor(sqrt(m))*t
T = 100  # number of parties*number of trees trained in parallel, which determines the number of inputs shared in parallel, and thus the number of iterations for the benchmark of input sharing. This is needed to set the number of iterations for the benchmark, which should be representative of the actual number of inputs shared in a real training scenario. The number of inputs shared depends on the number of parties, trees trained in parallel, samples, attributes, attribute values, labels, and depth of the tree. For example, for 10 parties training 10 trees in parallel, T=100; for 3 parties training 5 trees in parallel, T=15; etc..
sqm = math.floor(math.sqrt(m))
S1 = 2 * T * alpha * n * m
S2 = 4 * T * (2**d - 1) * alpha * sqm * t * n
S3 = T * (2**d - 1) * n * t
S4 = T * (2**d - 1) * alpha * sqm * t
S  = S1 + S2 + S3 + S4

print_ln("Sharing benchmark: S1=%s, S2=%s, S3=%s, S4=%s, S_total=%s",
         S1, S2, S3, S4, S)
array_a = types.personal(0, Array(S, types.cfix))
@for_range_opt(S)
def _(i):
    value = cint(42)
    value_cfix =  types.cfix(value)
    array_a[i] = value_cfix
start_timer(5)
@for_range_opt_multithread(n_threads, S)
def _(i):
    array_sa = sfix.Array(S)
    array_sa[:] = sfix(array_a[:])
stop_timer(5)