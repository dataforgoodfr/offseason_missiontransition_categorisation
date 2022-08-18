"""Unit tests for data_preparation """
from data_preparation import DataPreparation


# pylint: disable=missing-function-docstring
def test_data_preparation():
    aid0 = "<p>première aide: faire du recyclage. </p>"\
        "Le bénéficiaire fait du recyclage."

    dtp = DataPreparation()
    tokens = dtp.tokenize([aid0])
    print("Result: ", tokens)
    expected_result = ['premier', 'aide', 'faire', 'de', 'recyclage',
                       'le', 'bénéficiaire', 'faire', 'de', 'recyclage']
    assert tokens[0] == expected_result
