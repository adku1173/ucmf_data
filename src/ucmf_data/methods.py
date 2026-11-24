"""Shared timing and Herold evaluation for reconstruction methods."""

from time import perf_counter

import acoular as ac


def herold_metrics(parameters, data, sourcemap):
    """Evaluate maps (F, N) against source strengths (F, J) at the true locations.

    Use adaptive circular sectors, allowing multiple assignment, as in the old
    evaluation. metric_radius is in metres. All returned level errors are in dB.
    """
    evaluator = ac.tools.MetricEvaluator(
        data=sourcemap,
        target_data=data['source_strength'],
        grid=ac.ImportGrid(pos=parameters.focus_grid),
        target_grid=ac.ImportGrid(pos=parameters.source_locations),
        sector=ac.CircSector(r=parameters.metric_radius, include_border=False),
        adaptive_sector_size=True,
        multi_assignment=True,
    )
    return {
        'specific_level_error': evaluator.get_specific_level_error(),
        'overall_level_error': evaluator.get_overall_level_error(),
        'inverse_level_error': evaluator.get_inverse_level_error(),
    }


def evaluate_method(calculate, parameters, data):
    """Time one method call, then add shared Herold metrics without mutating its result.

    calculate(parameters, data) must return a dict containing sourcemap (F, N).
    Other fields, including nested solver information, are retained unchanged.
    time and the three level-error keys belong to this wrapper. Methods must
    not modify the shared inputs in data.

    time is wall-clock seconds for the entire call across all frequencies,
    excluding dependency generation and metric evaluation. No implicit warm-up
    or second solver call is performed; prewarm methods before benchmarking if
    compilation/setup should be excluded. FLOPs are not collected.
    """
    start = perf_counter()
    result = calculate(parameters, data)
    elapsed = perf_counter() - start
    return result | {'time': elapsed} | herold_metrics(parameters, data, result['sourcemap'])
