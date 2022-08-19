"""Unit tests for data_preparation """
import numpy as np
from baseline_model import BaselineClassifier


# pylint: disable=missing-function-docstring
def test_baseline_classifier():
    tokens = ['foo', 'bar']

    clf = BaselineClassifier("topic", tokens)
    X = np.array([['le', 'chat'], [],
                  ['bar', 'z', 'chien']], dtype=list)
    y_pred = clf.predict(X)
    assert (y_pred == [False, False, True]).all()
