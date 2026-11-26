"""Finite-horizon plurality success kernels for the native count CRP.

The correct-answer hypothesis and q/theta stay fixed across a continuation.
Wrong-draw allocations are summed with truncated exponential generating
functions. Only counts and first-appearance ordering enter the calculation.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb,factorial
import numpy as np


@lru_cache(None)
def cycles(size,groups,cap):
    """Permutations of size labelled items into groups cycles of length <=cap."""
    if size==0:return int(groups==0)
    if groups<=0 or cap<=0:return 0
    return sum(comb(size-1,s-1)*factorial(s-1)*cycles(size-s,groups-1,cap)
               for s in range(1,min(size,cap)+1))


@lru_cache(None)
def new_group_egf(w,cap,unseen):
    """[x^n theta^j] CRP new groups, with unseen-correct tie averaging."""
    out=np.zeros((w+1,w+1))
    for n in range(w+1):
        for j in range(n+1):
            if not unseen:
                out[n,j]=cycles(n,j,cap)/factorial(n)
            else:
                # New wrong groups tying the unseen correct answer all have
                # size cap. The correct answer arrives first among r+1 equal-
                # sized groups with probability 1/(r+1).
                for r in range(min(j,n//cap)+1):
                    rest=n-r*cap
                    out[n,j]+=cycles(rest,j-r,cap-1)/factorial(rest)/(cap**r*factorial(r)*(r+1))
    out.setflags(write=False)
    return out


@lru_cache(None)
def allocation_coefficient(early,late,limit,w,unseen):
    """Ordered wrong-draw weights, by number of newly created wrong groups."""
    old=np.zeros(w+1);old[0]=1.
    for counts,cap in ((early,limit-1),(late,limit)):
        for n in counts:
            if n>cap:return tuple(np.zeros(w+1))
            factor=np.array([comb(n+r-1,r) if r<=cap-n else 0. for r in range(w+1)])
            old=np.convolve(old,factor)[:w+1]
    new=new_group_egf(w,limit,unseen)
    return tuple(factorial(w)*sum(old[r]*new[w-r,j] for r in range(w+1)) for j in range(w+1))


@lru_cache(None)
def success_terms(nc,early,late,horizon,unseen):
    """(correct additions, wrong additions, new wrong groups, coefficient)."""
    result=[]
    for correct in range(horizon+1):
        limit=nc+correct
        if limit==0:continue
        wrong=horizon-correct
        coefficients=allocation_coefficient(early,late,limit,wrong,unseen)
        for groups,value in enumerate(coefficients):
            if value>0:result.append((correct,wrong,groups,comb(horizon,correct)*value))
    return tuple(result)


def hypothesis_signature(counts,first,hypothesis):
    nc=int(counts[hypothesis]) if hypothesis<16 else 0
    unseen=hypothesis==16
    first_c=first[hypothesis] if not unseen else 1000
    early=tuple(sorted(int(n) for k,n in enumerate(counts) if n>0 and k!=hypothesis and first[k]<first_c))
    late=tuple(sorted(int(n) for k,n in enumerate(counts) if n>0 and k!=hypothesis and first[k]>=first_c))
    return nc,early,late,unseen


def conditional_success(counts,first,hypothesis,horizon,q,theta):
    """NumPy reference for tests; vectorized in scalar or array q/theta."""
    nc,early,late,unseen=hypothesis_signature(counts,first,hypothesis)
    m=sum(early)+sum(late)
    total=np.zeros(np.broadcast(q,theta).shape)
    for correct,wrong,groups,coefficient in success_terms(nc,early,late,horizon,unseen):
        denom=np.ones_like(total)
        for r in range(wrong):denom=denom*(theta+m+r)
        total=total+coefficient*q**correct*(1-q)**wrong*theta**groups/denom
    return total
