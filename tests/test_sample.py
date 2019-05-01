import unittest
import Shell as sh
import numpy as np
import pandas as pd

def test_view():
    s = sh.Shell(1)
    result = s.view_data()
    assert result == 0


    