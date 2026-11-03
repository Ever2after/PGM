import numpy as np
from scipy.special import betainc

from pgm_asc.policies.baselines import trace
from pgm_asc.policies.baselines import cges_distribution


def features(n, **fields):
    return [{"lns_geom": .8, "lns_arith": .8, "reasc_bottom10_group128": 10., **fields} for _ in range(n)]


def test_asc_beta_probability_and_first_occurrence_ties():
    result = trace(["1", "2", "1"], features(3), "asc")
    assert result["chosen"].tolist() == [0, 0, 0]
    assert np.allclose(result["confidence"], [1-betainc(2,1,.5), .5, 1-betainc(3,2,.5)])


def test_esc_uses_disjoint_windows_and_returns_unanimous_window():
    answers = ["1"]*3 + ["2"]*5
    result = trace(answers, features(8), "esc", {"window": 4})
    assert not result["confidence"][:7].any()
    assert result["confidence"][7] == 1
    assert result["chosen"][7] == 1


def test_cges_dynamic_support_matches_direct_likelihood():
    ids, confidence = [0, 0, 1], [.8, .7, .9]
    expected = []
    for label in range(3):
        expected.append(np.prod([c if i==label else (1-c)/2 for i,c in zip(ids,confidence)]))
    expected = np.asarray(expected)/sum(expected)
    assert np.allclose(cges_distribution(ids,confidence,2), expected)
    assert np.isclose(cges_distribution([0],[.8],1)[0],.8)


def test_reasc_gate_and_unit_weight_reduce_to_asc():
    f=features(3)
    settings={"mean":10.,"std":1.,"lambda":.7,"gate":None}
    a=trace(["1","2","1"],f,"asc")
    b=trace(["1","2","1"],f,"reasc",settings)
    assert np.allclose(a["confidence"],b["confidence"])
    settings["gate"]=9.
    assert trace(["1"],features(1),"reasc",settings)["confidence"][0] == 1


def test_baseline_decisions_do_not_observe_future_answers():
    for method, settings in [("asc",{}),("esc",{"window":4}),("cges",{"confidence":"lns_geom"})]:
        a=trace(["1","2"],features(2),method,settings)
        b=trace(["1","2","3","3"],features(4),method,settings)
        assert np.allclose(a["confidence"],b["confidence"][:2])
        assert np.array_equal(a["chosen"],b["chosen"][:2])
