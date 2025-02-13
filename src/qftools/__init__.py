"""Quant Finance Tools
By Alex Kaufman el-diez@yandex.ru

Collection of useful python procedures"""

from .simulations import (
    generate_bm,
    generate_gbm,
    generate_uo
)
from .timeseries import (
    LinearRegression,
    AugmentedDickeyFuller,
    EngleGranger,
    fit_uo_params,
    mle_uo_params
)
from .vol_estimates import (
    ctc_vol,
    parkinson,
    rodgers_satchell,
    garman_klass,
    yang_zhang
)
from .options import (
    bsm_value,
    bsm_delta,
    bs76_value,
    bsm_vega,
    bsm_ivol,
    bsm_theta,
    bsm_gamma,
    bsm_rho,
    bsm_delta_forward,
    bsm_delta_prem,
    implied_foreign_depo,
    implied_domestic_depo,
    atm_dns,
    strike_from_delta_and_vol,
    market_smile,
    VanillaOption

)
__version__ = '0.1.0'

__all__ = [
    'generate_bm',
    'generate_gbm',
    'generate_uo',
    'ctc_vol',
    'parkinson',
    'rodgers_satchell',
    'garman_klass',
    'yang_zhang',
    'LinearRegression',
    'AugmentedDickeyFuller',
    'EngleGranger',
    'fit_uo_params',
    'mle_uo_params',
    'bsm_value',
    'bsm_delta',
    'bs76_value',
    'bsm_vega',
    'bsm_ivol',
    'bsm_theta',
    'bsm_gamma',
    'bsm_rho',
    'bsm_delta_forward',
    'bsm_delta_prem',
    'implied_foreign_depo',
    'implied_domestic_depo',
    'atm_dns',
    'strike_from_delta_and_vol',
    'market_smile',
    'VanillaOption'
]