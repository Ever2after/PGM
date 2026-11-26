"""Exact transition kernels must match the shared native CRP simulator."""
import numpy as np
import pytest

from pgm_asc.p1_transitions.kernel import conditional_success
from pgm_asc.diagnostics.predictive import simulate


def brute_success(counts,first,c,h,q,theta):
    """Enumerate labelled continuations, only for h<=3 in tests."""
    def recurse(counts,first,next_wrong,remaining):
        if remaining==0:
            leader=np.where(counts==counts.max(),first,1000).argmin()
            return float(leader==c)
        total=0.;step=h-remaining+1
        wrong=counts.copy();wrong[c]=0
        m=wrong.sum()
        choices=[(c,q)]
        choices += [(k,(1-q)*n/(m+theta)) for k,n in enumerate(wrong) if n>0]
        choices += [(next_wrong,(1-q)*theta/(m+theta))]
        for k,prob in choices:
            cc,ff=counts.copy(),first.copy();cc[k]+=1;ff[k]=min(ff[k],step)
            total+=prob*recurse(cc,ff,next_wrong+(k==next_wrong),remaining-1)
        return total
    cc=np.zeros(32);cc[:16]=counts
    ff=np.full(32,1000);ff[:16]=first
    return recurse(cc,ff,17,h)


@pytest.mark.parametrize('short_counts',[[1],[2,1],[1,2],[1,1,1],[]])
@pytest.mark.parametrize('h',[1,2,3])
def test_exact_kernel_matches_bruteforce_including_unseen_correct_and_ties(short_counts,h):
    counts=np.r_[short_counts,np.zeros(16-len(short_counts))]
    first=np.r_[np.arange(len(short_counts))-16,np.full(16-len(short_counts),1000)]
    for c in list(range(len(short_counts)))+[16]:
        for q,theta in ((.2,.3),(.7,2.)):
            expected=brute_success(counts,first,c,h,q,theta)
            actual=conditional_success(counts,first,c,h,q,theta)
            assert float(actual)==pytest.approx(expected,abs=2e-12)


def test_horizon_eight_success_matches_high_draw_native_monte_carlo():
    counts=np.r_[2.,1.,1.,np.zeros(13)]
    first=np.r_[-4,-3,-1,np.full(13,1000)]
    n=120000;q=.35;theta=1.7
    for c in (0,1,16):
        expected=float(conditional_success(counts,first,c,8,q,theta))
        rng=np.random.default_rng(71+c)
        result=simulate(None,counts,first,None,None,[8],rng,n,True,
            latent_draws=(np.full(n,c,int),np.full(n,q),np.full(n,theta)))[8]
        observed=result['future_reference_correct']
        assert abs(expected-observed)<5*np.sqrt(max(expected*(1-expected),1e-4)/n)


def dynamic_success(counts,first,c,h,q,theta):
    """Independent finite-state DP, with no generating-function coefficients."""
    from functools import lru_cache
    unseen=c==16
    nc=int(counts[c]) if not unseen else 0
    wrong=tuple(int(n) for k,n in enumerate(counts) if n>0 and k!=c)
    early=tuple(bool(unseen or first[k]<first[c]) for k,n in enumerate(counts) if n>0 and k!=c)
    @lru_cache(None)
    def solve(old,new,nc,rank,remaining):
        if remaining==0:
            return float(nc>0 and all(n<=nc-int(e) for n,e in zip(old,early)) and
                all(n<=nc-int(unseen and j<rank) for j,n in enumerate(new)))
        value=q*solve(old,new,nc+1,len(new) if unseen and nc==0 else rank,remaining-1)
        denom=sum(old)+sum(new)+theta
        for j,n in enumerate(old):
            nxt=list(old);nxt[j]+=1
            value+=(1-q)*n/denom*solve(tuple(nxt),new,nc,rank,remaining-1)
        for j,n in enumerate(new):
            nxt=list(new);nxt[j]+=1
            value+=(1-q)*n/denom*solve(old,tuple(nxt),nc,rank,remaining-1)
        value+=(1-q)*theta/denom*solve(old,new+(1,),nc,rank,remaining-1)
        return value
    return solve(wrong,(),nc,-1,h)


@pytest.mark.parametrize('c',[0,1,16])
@pytest.mark.parametrize('q,theta',[(.1,.3),(.35,1.7),(.8,5.)])
def test_horizon_eight_kernel_matches_independent_exact_dynamic_program(c,q,theta):
    counts=np.r_[2.,1.,1.,np.zeros(13)]
    first=np.r_[-4,-3,-1,np.full(13,1000)]
    expected=dynamic_success(counts,first,c,8,q,theta)
    actual=float(conditional_success(counts,first,c,8,q,theta))
    assert actual==pytest.approx(expected,abs=3e-12,rel=3e-12)
