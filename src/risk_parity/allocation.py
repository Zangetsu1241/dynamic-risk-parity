"""Long-only risk budgeting with numerical safeguards."""
import numpy as np
from scipy.optimize import minimize


def validate_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("covariance must be square with at least two assets")
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("covariance must be finite and symmetric")
    if np.min(np.linalg.eigvalsh(cov)) <= 0:
        raise ValueError("covariance must be positive definite")
    return cov


def regularized_covariance(returns, shrinkage=0.15, floor=1e-8):
    """Sample covariance shrunk to its diagonal; causal when input is historical only."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("returns must be finite T x N, with T>=3 and N>=2")
    if not 0 <= shrinkage <= 1 or floor <= 0:
        raise ValueError("shrinkage must be in [0,1] and floor positive")
    s = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    cov = (1-shrinkage)*s + shrinkage*np.diag(np.diag(s))
    cov = cov + np.eye(cov.shape[0])*floor
    return validate_covariance(cov)


def risk_contributions(weights, cov):
    """Absolute Euler decomposition of portfolio volatility."""
    c = validate_covariance(cov)
    w = np.asarray(weights, dtype=float)
    if w.shape != (c.shape[0],) or not np.isfinite(w).all():
        raise ValueError("invalid weight vector")
    variance = float(w @ c @ w)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    return w*(c @ w)/np.sqrt(variance)


def inverse_volatility(cov):
    c = validate_covariance(cov)
    inv = 1/np.sqrt(np.diag(c))
    return inv/inv.sum()


def equal_risk_contribution(cov, tolerance=1e-7, max_iter=2000):
    """Find long-only fully invested weights with near-equal volatility contributions.

    Uses log-weight risk budgeting objective, then explicitly verifies risk budgets.
    """
    c = validate_covariance(cov)
    n = c.shape[0]
    # Convex formulation: minimize 0.5*x^T C x - sum(log x), then normalize.
    # Gradient analytical; log-barrier prevents boundary convergence.
    scale = c / np.mean(np.diag(c))
    def fun(z):
        x=np.exp(np.clip(z,-30,30))
        return 0.5*(x@scale@x) - np.sum(z)
    def jac(z):
        x=np.exp(np.clip(z,-30,30))
        return x*(scale@x)-1
    res=minimize(fun,np.zeros(n),jac=jac,method='BFGS',options={'gtol':1e-10,'maxiter':max_iter})
    w=np.exp(np.clip(res.x,-30,30))
    w=w/w.sum()
    rc=risk_contributions(w,c)
    budgets=rc/rc.sum()
    if not np.isfinite(w).all() or np.max(np.abs(budgets-1/n)) > tolerance:
        raise RuntimeError(f"ERC optimization failed budget tolerance: {np.max(np.abs(budgets-1/n)):.3g}")
    return w


def minimum_variance(cov):
    c=validate_covariance(cov)
    n=len(c)
    result=minimize(lambda w: float(w@c@w), np.full(n,1/n),
                    jac=lambda w: 2*(c@w),
                    method='SLSQP',bounds=[(0,1)]*n,
                    constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1,'jac':lambda w: np.ones_like(w)}],
                    options={'ftol':1e-12,'maxiter':2000})
    if not result.success:
        raise RuntimeError('minimum variance optimization failed: '+result.message)
    w=np.maximum(result.x,0)
    return w/w.sum()
