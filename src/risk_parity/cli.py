"""Research command line entrypoint."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import simulated_prices,load_prices
from .backtest import walk_forward,Config,metrics


def main(argv=None):
    p=argparse.ArgumentParser(description='Dynamic risk-parity portfolio research')
    p.add_argument('--csv',help='date plus >=2 adjusted-close asset columns')
    p.add_argument('--days',type=int,default=850)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lookback',type=int,default=126)
    p.add_argument('--rebalance-every',type=int,default=21)
    p.add_argument('--cost-bps',type=float,default=5)
    p.add_argument('--shrinkage',type=float,default=.15)
    p.add_argument('--output',default='results')
    a=p.parse_args(argv)
    prices=load_prices(a.csv) if a.csv else simulated_prices(a.days,a.seed)
    config=Config(lookback=a.lookback,rebalance_every=a.rebalance_every,cost_bps=a.cost_bps,shrinkage=a.shrinkage)
    result,weights=walk_forward(prices,config)
    report={s:metrics(group.net_return,group.turnover) for s,group in result.groupby('strategy',sort=True)}
    info={'data_type':'user_csv' if a.csv else 'simulated','config':config.__dict__,
          'assumptions':'signals use close t and the following close-to-close return; same-close trading is a research approximation, not executable proof',
          'strategies':report}
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'returns.csv')
    weights.to_csv(out/'weights.csv')
    (out/'metrics.json').write_text(json.dumps(info,indent=2))
    fig,ax=plt.subplots(figsize=(11,6))
    for s,group in result.groupby('strategy'):
        group=(1+group.net_return).cumprod()
        ax.plot(group.index,group.values,label=s)
    ax.legend()
    ax.set_title('Out-of-sample wealth (1.0 initial, proportional costs)')
    ax.set_ylabel('Portfolio wealth')
    fig.tight_layout()
    fig.savefig(out/'equity_curves.png',dpi=150)
    plt.close(fig)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
