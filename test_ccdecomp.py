from ccdecomp import CostModel
import pandas as pd
import numpy as np

def test_title():
    cp = CostModel(title='Title', equation='x1 x2 + x3 x4')
    assert cp.title == 'Title'

def test_regularize_equation():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._equation == 'x1 x2 + x3 x4'

    cp = CostModel('x1 *x2 + x3 x4')
    assert cp._equation == 'x1 x2 + x3 x4'

    cp = CostModel('  x1 *x2   + x3 x4  ')
    assert cp._equation == 'x1 x2 + x3 x4'

    cp = CostModel('x1*x2+x3*x4')
    assert cp._equation == 'x1 x2 + x3 x4'

def test_parse_cost_components():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._cost_component_exprs == ['x1 x2', 'x3 x4']

    cp = CostModel('x1*x2 + x3 x4')
    assert cp._cost_component_exprs == ['x1 x2', 'x3 x4']

    cp = CostModel('x1* x2 + x3 x4')
    assert cp._cost_component_exprs == ['x1 x2', 'x3 x4']

    cp = CostModel('x1*x2+x3*x4')
    assert cp._cost_component_exprs == ['x1 x2', 'x3 x4']

def test_count_cc_terms():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._n_components == 2

def test_cc_names():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._cost_component_names == ['C1', 'C2']

def test_gather_symbols():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._symbols == ['x1', 'x2', 'x3', 'x4']

def test_bind_data():
    import pandas as pd
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [1,1]
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    assert cp.data.loc[1,'x1'] == 1
    assert cp.data.loc[1,'x2'] == 2
    assert cp.data.loc[1,'x3'] == 3
    assert cp.data.loc[1,'x4'] == 4

def test_ccdecomposition_no_change():
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [1,1]
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    time_spans = [(1,2)]
    cp.cost_change_decomposition(time_spans)
    elements_equal = \
        cp.cost_change_contributions.to_numpy() \
        == np.zeros((1,2 + cp._n_components + cp._n_variables))
    assert elements_equal.all()

def test_ccdecomposition_starts_at_zero():
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [0,1] #<----
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    time_spans = [(1,2)]
    cp.cost_change_decomposition(time_spans)
    ccd_table = cp.cost_change_contributions.to_numpy().flatten().astype(float)
    correct_ccd_table = np.array([2, 2, 2, 0, 2, 0, 0, 0], dtype=np.float64)
    assert np.allclose(ccd_table, correct_ccd_table)

def test_ccdecomposition_ends_at_zero():
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [1,0]  #<----
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    time_spans = [(1,2)]
    cp.cost_change_decomposition(time_spans)
    ccd_table = cp.cost_change_contributions.to_numpy().flatten().astype(float)
    correct_ccd_table = np.array([-2, -2, -2, 0, -2, 0, 0, 0], dtype=np.float64)
    assert np.allclose(ccd_table, correct_ccd_table)

def test_varchanges_starts_at_zero():
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [0,1]  #<----
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    time_spans = [(1,2)]
    cp.cost_change_decomposition(time_spans)
    var_changes = cp.variable_changes.to_numpy().flatten().astype(float)
    correct_var_changes = np.array([np.inf, 0.0, 0.0, 0.0])
    assert np.allclose(var_changes, correct_var_changes)

def test_varchanges_ends_at_zero():
    cp = CostModel('x1 x2 + x3 x4')
    time = [1,2]
    data = pd.DataFrame(index=time)
    data['x1'] = [1,0]  #<----
    data['x2'] = [2,2]
    data['x3'] = [3,3]
    data['x4'] = [4,4]
    cp.bind_data(data)
    time_spans = [(1,2)]
    cp.cost_change_decomposition(time_spans)
    var_changes = cp.variable_changes.to_numpy().flatten().astype(float)
    correct_var_changes = np.array([-np.inf, 0.0, 0.0, 0.0])
    assert np.allclose(var_changes, correct_var_changes)

# todo: add tests involving zeros, infinities, etc.

# todo: do we do checks that parameters stay the same in all periods, as promised in the identify_parameters() method?

#todo: add tests for the behavior of log changes in variables

#todo: test that the sum of the variable changes = sum of cost component change = total change computed directly

#todo: create a special class that acts like a zero in same contests and like a tiny number elsewhere
#https://stackoverflow.com/questions/25022079/extend-python-build-in-class-float

#todo: add tests that the program gives various run-time errors if it's misused