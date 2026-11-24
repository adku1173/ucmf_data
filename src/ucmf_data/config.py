"""acoupipe configuration with the default CMF features."""

from functools import partial

from acoupipe.datasets import Config as AcoupipeConfig
from acoupipe.datasets.features import Feature, create_feature, remove_unrequested

from ucmf_data.features import (
    analytic_reference_gain,
    analytic_transfer,
    cleansc,
    csm,
    measured_transfer,
    sensing_matrix,
    source_strength,
    x,
    y,
)
from ucmf_data.methods import evaluate_method
from ucmf_data.parameters import MiracleA1Parameters, Parameters
from ucmf_data.samplers import attach_samplers

import numpy as np


class Config(AcoupipeConfig):
    """Provide default scene samplers and source_strength, x, csm, y, A and cleansc by name.

    Only requested outputs are retained. x and csm depend on source_strength;
    y and cleansc reuse csm. A depends only on the ParameterSet. Dependencies
    are shared regardless of request order. Frequencies and processing
    settings always come from the current ParameterSet.
    MIRACLE parameters select measured transfer, synthetic parameters analytic.
    source_strength and transfer are referenced to the nominal ref; measured
    transfer is read once per case and also yields the reference gains.
    """

    def __init__(self, parameters=None, **traits):
        super().__init__(parameters=parameters, **traits)
        self.create_samplers()

    def create_parameters(self):
        return Parameters()

    def create_samplers(self):
        """Register scene samplers once; override for custom sampling or fixed scenes."""
        attach_samplers(self)

    def _transfer(self, data):
        if 'transfer' not in data:
            transfer = measured_transfer if isinstance(self.parameters, MiracleA1Parameters) else analytic_transfer
            data['transfer'], data['reference_gain'] = transfer(self.parameters)

    def _source_strength(self, data):
        if 'reference_gain' not in data:
            if isinstance(self.parameters, MiracleA1Parameters):
                self._transfer(data)
            else:
                data['reference_gain'] = analytic_reference_gain(self.parameters)
        if 'source_strength' not in data:
            data['source_strength'] = source_strength(self.parameters, data)

    def _csm(self, data):
        self._source_strength(data)
        self._transfer(data)
        if 'csm' not in data:
            data['csm'] = csm(self.parameters, data)

    def method_feature(self, name, calculate):
        """Create a requested method feature with timing, Herold metrics and solver extras.

        calculate(parameters, data) returns a dict with sourcemap (F, N) and
        arbitrary extra fields. Shared CSM and source strengths are prepared
        outside the timed call, once per case regardless of request order.
        Pass the returned Feature in Dataset.generate(features=[...]).

        Nested values currently work with generate(), not acoupipe's HDF5 or
        TensorFlow exporters. dtype=dict describes the container, not its leaves.
        Default array features remain unchanged until nested export is supported.
        """

        def feature_func(sampler, data):
            self._csm(data)
            if name not in data:
                data[name] = evaluate_method(calculate, self.parameters, data)
            return {name: data[name]}

        return create_feature(feature_func, name=name, shape=(), dtype=dict)

    def _feature(self, name, calculate, shape, dtype=np.float64):
        def feature_func(sampler, data):
            if name in ('csm', 'y', 'cleansc'):
                self._csm(data)
            elif name != 'A':
                self._source_strength(data)
            if name not in data:
                data[name] = calculate(self.parameters, data)
            return {name: data[name]}

        return create_feature(feature_func, name=name, shape=shape, dtype=dtype)

    def _get_default_feature_source_strength(self, **_kwargs):
        return self._feature('source_strength', source_strength, (None, None))

    def _get_default_feature_x(self, **_kwargs):
        return self._feature('x', x, (None, None))

    def _get_default_feature_csm(self, **_kwargs):
        return self._feature('csm', csm, (None, None, None), np.complex128)

    def _get_default_feature_y(self, **_kwargs):
        return self._feature('y', y, (None, None))

    def _get_default_feature_A(self, **_kwargs):
        return self._feature('A', sensing_matrix, (None, None, None))

    def _get_default_feature_cleansc(self, **_kwargs):
        return self._feature('cleansc', cleansc, (None, None))

    def get_cleanup_func(self, features):
        keep = [feature.name if isinstance(feature, Feature) else feature for feature in features]
        return partial(remove_unrequested, keep=keep)
