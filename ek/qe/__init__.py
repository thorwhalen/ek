"""Reference-free quality estimation: the ``estimate_quality()`` side of ``ek``.

This package implements the online (no-gold) half of evaluation as the strict
pipeline ``signal -> calibrate -> validate -> decide`` (``misc/docs/ek_03``):

- **signals** (cheapest first): the deterministic :class:`~ek.qe.verifiers.VerifierSignal`
  (schema/checksum/cross-field, cost tier 1, runs always), free intrinsic
  :class:`~ek.qe.signals.IntrinsicConfidenceSignal` / :class:`~ek.qe.signals.LogprobSignal`
  (tier 2), and the flagship ROVER :class:`~ek.qe.rover.AgreementSignal` multi-engine
  vote (tier 3);
- **calibrate** -- :class:`~ek.qe.calibrate.PlattCalibrator` (default) / isotonic /
  temperature, with Mondrian grouping and ECE diagnostics: *non-optional before any
  gate*;
- **decide** -- :class:`~ek.qe.decide.CostSensitiveGate` (default),
  :class:`~ek.qe.decide.ConformalGate`, and :class:`~ek.qe.decide.RiskControlGate`,
  reading only calibrated probabilities.

Everything is pure-Python and dependency-free by default; the ``ek[calibration]``
(netcal/sklearn/MAPIE/crepes) and ``ek[agreement]`` (uqlm) extras supply
library-backed equivalents, imported lazily so ``import ek`` stays light.

Importing this package registers the built-in strategies (signals, aggregators,
checks, calibrators, policies) by name in :mod:`ek.registry`.
"""

from __future__ import annotations

from ..registry import register
from .calibrate import (
    GroupCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    load_calibrator,
    reliability_curve,
    save_calibrator,
    sklearn_calibrator,
)
from .decide import (
    ConformalGate,
    CostSensitiveGate,
    GroupConformalGate,
    RiskControlGate,
    risk_coverage_curve,
    split_conformal_quantile,
)
from .rover import AgreementSignal, RoverConsensus, RoverSlot, rover
from .signals import (
    IntrinsicConfidenceSignal,
    LogprobSignal,
    geo_mean,
    length_normalized,
    mean_prob,
    min_prob,
)
from .verifiers import (
    VerifierSignal,
    checksum_validator,
    enum_validator,
    isbn_check,
    iban_check,
    luhn_check,
    range_validator,
    regex_validator,
    schema_validator,
    totals_consistent,
)

__all__ = [
    # ROVER (#3)
    "rover",
    "RoverConsensus",
    "RoverSlot",
    "AgreementSignal",
    # signals (#5)
    "VerifierSignal",
    "LogprobSignal",
    "IntrinsicConfidenceSignal",
    "geo_mean",
    "length_normalized",
    "min_prob",
    "mean_prob",
    # calibration (#5)
    "PlattCalibrator",
    "IsotonicCalibrator",
    "TemperatureCalibrator",
    "GroupCalibrator",
    "expected_calibration_error",
    "reliability_curve",
    "save_calibrator",
    "load_calibrator",
    "sklearn_calibrator",
    # decision (#5)
    "CostSensitiveGate",
    "ConformalGate",
    "GroupConformalGate",
    "RiskControlGate",
    "risk_coverage_curve",
    "split_conformal_quantile",
    # verifier helpers
    "checksum_validator",
    "regex_validator",
    "range_validator",
    "enum_validator",
    "schema_validator",
    "totals_consistent",
    "luhn_check",
    "iban_check",
    "isbn_check",
]

# Register signal strategies by name (open-closed; third parties add via entry points).
register("signals", "agreement", AgreementSignal)
register("signals", "verifier", VerifierSignal)
register("signals", "logprob", LogprobSignal)
register("signals", "intrinsic", IntrinsicConfidenceSignal)
