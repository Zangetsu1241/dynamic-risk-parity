"""Chronological, periodic close-to-close portfolio experiment."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .allocation import equal_risk_contribution, inverse_volatility, minimum_variance, regularized_covariance

@dataclass(frozen=True)
class Config:
    lookback:int=126
    rebalance_every:int=21
    cost_bps:float=5.0
    shrinkage:float=.15
    min_train:int=63


def walk_forward(prices, config=Config()):
    """At close t use only returns <=t, then evaluate portfolio on (t,t+1].

    Portfolio return uses weighted simple asset returns and weights drift between rebalances.
    Transaction costs are deducted from portfolio simple return on the first period after trade.
    """
    if config.lookback<3 or config.min_train<3 or config.rebalance_every<1 or config.cost_bps<0 or not 0<=config.shrinkage<=1:
        raise ValueError('invalid configuration')
    if not isinstance(prices,pd.DataFrame) or prices.shape[1]<2 or len(prices)<config.min_train+3:
        raise ValueError('prices must be a DataFrame with 2+ assets and enough dates')
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('index must be strictly increasing')
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices.to_numpy(dtype=float)<=0).any():
        raise ValueError('prices must be positive finite')
    rets=prices.pct_change().dropna()
    n=len(rets.columns)
    strategies=['equal_weight','inverse_volatility','risk_parity','minimum_variance']
    portfolio={s:np.full(n,1/n) for s in strategies}
    record=[]
    weights=[]
    start=max(config.min_train,config.lookback)
    # Row t's observed return is from t-1 to t; feature window through t is known at close t.
    for t in range(start,len(rets)-1):
        is_rebalance=(t-start)%config.rebalance_every==0
        next_ret=rets.iloc[t+1].to_numpy(dtype=float)
        for name in strategies:
            held=portfolio[name]
            turnover=0.0
            if is_rebalance:
                window=rets.iloc[t-config.lookback+1:t+1].to_numpy(dtype=float)
                cov=regularized_covariance(window,config.shrinkage)
                target={'equal_weight':lambda:np.full(n,1/n),
                        'inverse_volatility':lambda:inverse_volatility(cov),
                        'risk_parity':lambda:equal_risk_contribution(cov),
                        'minimum_variance':lambda:minimum_variance(cov)}[name]()
                turnover=float(np.abs(target-held).sum())
                held=target
            gross=float(held@next_ret)
            net=gross-turnover*config.cost_bps/10000
            if net<=-1:
                raise ValueError('net portfolio return at or below -100%')
            portfolio[name]=held*(1+next_ret)/(1+gross) # drift from return on the invested portfolio
            record.append({'date':rets.index[t+1],'signal_date':rets.index[t],
                           'strategy':name,'gross_return':gross,'net_return':net,
                           'turnover':turnover,'rebalance':is_rebalance})
            weights.append({'date':rets.index[t+1],'strategy':name,**dict(zip(rets.columns,held))})
    return pd.DataFrame(record).set_index('date'),pd.DataFrame(weights).set_index('date')


def metrics(returns,turnover=None,periods=252):
    x=np.asarray(returns,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or np.any(x<=-1):
        raise ValueError('returns must be finite simple returns > -100%')
    equity=np.cumprod(1+x)
    peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
    annvol=float(x.std(ddof=1)*np.sqrt(periods)) if len(x)>1 else 0.0
    sharpe=float(np.mean(x)/x.std(ddof=1)*np.sqrt(periods)) if len(x)>1 and x.std(ddof=1)>0 else 0.0
    ret={'total_return':float(equity[-1]-1),
         'annualized_return':float(equity[-1]**(periods/len(x))-1),
         'annualized_volatility':annvol,'sharpe':sharpe,
         'max_drawdown':float(np.min(equity/peak-1))}
    if turnover is not None:ret['total_turnover']=float(np.asarray(turnover).sum())
    return ret
