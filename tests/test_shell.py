import unittest
import Shell as sh
import numpy as np
import pandas as pd

s = sh.Shell(1)

def test_viewEmpty():
    result = s.view_data()
    assert result == 0


def test_viewPopulated():
    s.import_data("tests/testData.csv")
    result = s.view_data()
    assert result == 1

def test_viewPopulatedRows():
    s.clear()
    s.import_data("tests/testData.csv")
    result = s.view_data(5)
    assert result == 1


    