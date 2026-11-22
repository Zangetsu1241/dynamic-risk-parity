import numpy as np
import pandas as pd
import pytest
from risk_parity.allocation import (regularized_covariance, equal_risk_contribution,
                                    inverse_volatility, minimum_variance,risk_contributions)
from risk_parity.backtest import Config,walk_forward,metrics
from risk_parity.data import load_prices,simulated_prices


def test_diagonal_erc_matches_inverse_vol():
    c=np.diag([.01,.04,.09,.16])
    w=equal_risk_contribution(c)
    assert np.allclose(w,inverse_volatility(c),atol=2e-6)
    assert np.allclose(risk_contributions(w,c)/risk_contributions(w,c).sum(),.25,atol=1e-7)


def test_correlated_erc():
    c=np.array([[.04,.007,.009],[.007,.01,.003],[.009,.003,.02]])
    w=equal_risk_contribution(c)
    rc=risk_contributions(w,c)
    assert (w>0).all() and np.isclose(w.sum(),1)
    assert np.max(np.abs(rc/rc.sum()-1/3))<1e-7


def test_invariance_to_cov_scale():
    c=np.array([[.04,.007,.009],[.007,.01,.003],[.009,.003,.02]])
    assert np.allclose(equal_risk_contribution(c),equal_risk_contribution(100*c),atol=1e-7)


def test_invalid_covariance():
    with pytest.raises(ValueError):equal_risk_contribution([[1,2],[2,1]])
    with pytest.raises(ValueError):inverse_volatility([[1,np.nan],[np.nan,1]])


def test_minimum_variance_weights():
    c=np.diag([.04,.01,.09])
    w=minimum_variance(c)
    assert np.isclose(w.sum(),1) and np.min(w)>=0
    assert w@c@w < np.ones(3)@c@np.ones(3)/9


def test_regularized_covariance_positive_definite():
    x=np.repeat(np.arange(100,dtype=float)[:,None],3,axis=1)
    c=regularized_covariance(x)
    assert np.linalg.eigvalsh(c).min()>0


def test_backtest_alignment_and_costs():
    p=simulated_prices(265)
    result,weights=walk_forward(p,Config(lookback=65,rebalance_every=20))
    assert len(result)==len(weights)
    assert (result.signal_date<result.index).all()
    assert np.allclose(result.net_return,result.gross_return-result.turnover*.0005)
    assert np.isfinite(result.net_return).all()
    assert (result.turnover>=0).all()
    assert np.allclose(weights.iloc[:,1:].sum(axis=1),1)


def test_future_prices_cannot_affect_past_decisions():
    p=simulated_prices(280)
    changed=p.copy()
    changed.iloc[245:]*=np.array([1.9,.7,1.4,.5])
    cfg=Config(lookback=65,rebalance_every=20)
    a,wa=walk_forward(p,cfg)
    b,wb=walk_forward(changed,cfg)
    assert np.allclose(wa.loc[wa.index<p.index[245]].iloc[:,1:],wb.loc[wb.index<p.index[245]].iloc[:,1:])
    assert np.allclose(a.loc[a.index<p.index[245],'net_return'],b.loc[b.index<p.index[245],'net_return'])


def test_drift_without_rebalancing():
    dates=pd.bdate_range('2020-01-01',periods=105)
    p=pd.DataFrame({'a':100*1.002**np.arange(105),'b':100*1.001**np.arange(105)},index=dates)
    result, weights=walk_forward(p,Config(lookback=63,rebalance_every=100))
    w=weights[weights.strategy=='equal_weight']
    assert w.iloc[1]['a']>w.iloc[0]['a']


def test_metrics_zero():
    m=metrics(np.zeros(30))
    assert m['sharpe']==0 and m['total_return']==0 and m['max_drawdown']==0


def test_csv_validation(tmp_path):
    f=tmp_path/'bad.csv'
    f.write_text('date,a,b\n2022-01-03,100,101\n2022-01-03,101,102\n')
    with pytest.raises(ValueError): load_prices(f)


def test_invalid_configuration():
    with pytest.raises(ValueError):walk_forward(simulated_prices(110),Config(lookback=2))
