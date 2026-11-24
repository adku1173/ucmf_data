"""Generic method timing, Herold metrics and arbitrary solver information."""

from acoupipe.datasets import Dataset

from ucmf_data import Config, Parameters
from ucmf_data import config as config_module
from ucmf_data import methods as methods_module
from ucmf_data.features import cleansc, x
from ucmf_data.methods import evaluate_method, herold_metrics

import numpy as np
import pytest


class FixedConfig(Config):
    def create_samplers(self):
        pass


def scene(he=(8, 4)):
    parameters = Parameters(he=he, source_rms=np.sqrt(np.array([1.0, 2.0]) * 513), signal_seed=7)
    parameters.source_locations = parameters.focus_grid[:, [0, -1]]
    data = {'source_strength': np.tile([1.0, 2.0], (np.size(he), 1))}
    return parameters, data


@pytest.mark.parametrize('he', [4.0, [8.0, 4.0]])
def test_scaled_perfect_map_has_expected_herold_errors(he):
    parameters, data = scene(he)
    sourcemap = 2 * x(parameters, data)
    original = sourcemap.copy()
    metrics = herold_metrics(parameters, data, sourcemap)
    assert metrics['specific_level_error'].shape == (np.size(he), 2)
    assert metrics['overall_level_error'].shape == (np.size(he),)
    assert metrics['inverse_level_error'].shape == (np.size(he),)
    np.testing.assert_allclose(metrics['specific_level_error'], 10 * np.log10(2), atol=1e-12)
    np.testing.assert_allclose(metrics['overall_level_error'], 10 * np.log10(2), atol=1e-12)
    np.testing.assert_allclose(metrics['inverse_level_error'], 0, atol=1e-12)
    np.testing.assert_array_equal(sourcemap, original)


def test_timing_covers_exactly_one_solver_call_and_excludes_metrics(monkeypatch):
    parameters, data = scene()
    events = []
    times = iter([10.0, 10.25])
    solver_result = {
        'sourcemap': x(parameters, data).astype(np.float32),
        'iterations': np.array([5, 12]),
        'solver_info': {'converged': np.array([True, False]), 'residuals': [0.1, 0.2]},
    }

    def clock():
        events.append('clock')
        return next(times)

    def calculate(p, d):
        events.append('solve')
        assert p is parameters
        assert d is data
        return solver_result

    original = herold_metrics

    def metrics(p, d, sourcemap):
        events.append('metrics')
        return original(p, d, sourcemap)

    monkeypatch.setattr(methods_module, 'perf_counter', clock)
    monkeypatch.setattr(methods_module, 'herold_metrics', metrics)
    result = evaluate_method(calculate, parameters, data)
    assert events == ['clock', 'solve', 'clock', 'metrics']
    assert result['time'] == 0.25
    assert result['sourcemap'] is solver_result['sourcemap']
    assert result['iterations'] is solver_result['iterations']
    assert result['solver_info'] is solver_result['solver_info']
    assert set(solver_result) == {'sourcemap', 'iterations', 'solver_info'}
    assert 'flops' not in result


@pytest.mark.parametrize('csm_first', [False, True])
def test_multiple_methods_share_csm_and_only_requested_outputs_are_retained(csm_first, monkeypatch):
    parameters, _ = scene()
    config = FixedConfig(parameters=parameters)
    csm_calls = []
    solver_calls = []
    original = config_module.csm

    def csm(p, data):
        csm_calls.append(p.signal_seed)
        return original(p, data)

    def calculate(p, data):
        solver_calls.append(data['csm'])
        return {'sourcemap': x(p, data), 'solver_info': {'iterations': np.array([2, 3])}}

    monkeypatch.setattr(config_module, 'csm', csm)
    first = config.method_feature('first', calculate)
    second = config.method_feature('second', calculate)
    requested = [config._get_default_feature_csm(), first, second] if csm_first else [first, second]
    results = list(Dataset(config=config).generate(features=requested, size=2, progress_bar=False))
    assert len(csm_calls) == 2
    assert len(solver_calls) == 4
    for index, result in enumerate(results):
        assert solver_calls[2 * index] is solver_calls[2 * index + 1]
        assert set(result) == {'idx', 'first', 'second', *(['csm'] if csm_first else [])}
        for name in ('first', 'second'):
            assert result[name]['time'] >= 0
            np.testing.assert_allclose(result[name]['specific_level_error'], 0, atol=1e-12)
            np.testing.assert_array_equal(result[name]['solver_info']['iterations'], [2, 3])
    _, metadata = config.get_features([first, second], f=None, num=0)
    assert [feature.name for feature in metadata] == ['first', 'second']
    assert all(feature.dtype is dict for feature in metadata)


def test_cleansc_wrapper_preserves_the_default_sourcemap():
    parameters, _ = scene()
    config = FixedConfig(parameters=parameters)
    dataset = Dataset(config=config)
    expected = next(dataset.generate(features=['cleansc'], size=1, progress_bar=False))['cleansc']

    def calculate(p, data):
        return {'sourcemap': cleansc(p, data)}

    feature = config.method_feature('cleansc', calculate)
    result = next(dataset.generate(features=[feature], size=1, progress_bar=False))['cleansc']
    np.testing.assert_array_equal(result['sourcemap'], expected)
    assert result['time'] >= 0
    assert result['specific_level_error'].shape == (2, 2)
    assert result['overall_level_error'].shape == (2,)
    assert result['inverse_level_error'].shape == (2,)


def test_unrequested_method_is_not_executed():
    parameters, _ = scene()
    config = FixedConfig(parameters=parameters)

    def calculate(p, data):
        pytest.fail('Unrequested method was evaluated')

    config.method_feature('unused', calculate)
    result = next(Dataset(config=config).generate(features=['he'], size=1, progress_bar=False))
    assert set(result) == {'idx', 'he'}


def test_metrics_use_the_current_radius_for_integration():
    parameters, data = scene(4.0)
    sourcemap = x(parameters, data)
    sourcemap[0, 1] = 1
    distance = np.linalg.norm(parameters.focus_grid[:, 0] - parameters.focus_grid[:, 1])
    parameters.metric_radius = 0.4 * distance
    narrow = herold_metrics(parameters, data, sourcemap)
    parameters.metric_radius = 1.1 * distance
    wide = herold_metrics(parameters, data, sourcemap)
    np.testing.assert_allclose(narrow['specific_level_error'], [[0, 0]], atol=1e-12)
    np.testing.assert_allclose(wide['specific_level_error'], [[10 * np.log10(2), 0]], atol=1e-12)
    assert narrow['inverse_level_error'][0] < 0
    np.testing.assert_allclose(wide['inverse_level_error'], 0, atol=1e-12)


def test_metric_radius_defaults_to_nominal_aperture_and_can_be_overridden():
    assert Parameters().metric_radius == 0.05 * Parameters().aperture
    assert Parameters(aperture=2.0).metric_radius == 0.1
    assert Parameters(aperture=2.0, metric_radius=0.02).metric_radius == 0.02
