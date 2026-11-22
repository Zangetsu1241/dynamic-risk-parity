"""Data generation and strict adjusted-close CSV ingestion."""
import numpy as np
import pandas as pd


def simulated_prices(days=1100, seed=42):
    if days < 100:
        raise ValueError('at least 100 days required')
    rng=np.random.default_rng(seed)
    names=['equity','bonds','gold','commodities']
    # Three regimes with changing drift, volatility, and inter-asset dependence.
    means=np.array([[.00045,.00012,.0001,.00018],[-.0012,.00035,.00055,-.0008],[.0007,-.0001,.00025,.0005]])
    vols=np.array([[.009,.003,.007,.012],[.021,.007,.012,.025],[.012,.006,.01,.018]])
    corrs=np.array([[[1,-.15,.05,.40],[-.15,1,.05,-.10],[.05,.05,1,.10],[.40,-.10,.10,1]],
                    [[1,.20,.35,.70],[.20,1,.10,.10],[.35,.10,1,.30],[.70,.10,.30,1]],
                    [[1,.05,.05,.55],[.05,1,-.05,-.10],[.05,-.05,1,.20],[.55,-.10,.20,1]]])
    p=np.array([[.99,.008,.002],[.015,.98,.005],[.006,.009,.985]])
    state=0
    x=np.zeros((days,len(names)))
    for t in range(days):
        state=int(rng.choice(3,p=p[state]))
        cov=np.outer(vols[state],vols[state])*corrs[state]
        x[t]=rng.multivariate_normal(means[state],cov)
    dates=pd.bdate_range('2018-01-01',periods=days)
    return pd.DataFrame(100*np.exp(np.cumsum(x,axis=0)),index=dates,columns=names)


def load_prices(path):
    df=pd.read_csv(path)
    if 'date' not in df.columns or len(df.columns)<3:
        raise ValueError('CSV requires date and at least two asset columns')
    idx=pd.to_datetime(df['date'],errors='raise',utc=True)
    if idx.isna().any() or idx.duplicated().any() or not idx.is_monotonic_increasing:
        raise ValueError('dates must be nonmissing, unique and increasing')
    prices=df.drop(columns=['date']).apply(pd.to_numeric,errors='raise')
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any():
        raise ValueError('prices must be positive finite numbers, no missing values')
    prices.index=idx
    return prices
