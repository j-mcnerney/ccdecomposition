from ccdecomp import CostModel

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

def test_ccdecomposition1():
    import pandas as pd
    from numpy import zeros
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
        == zeros((1,2 + cp._n_components + cp._n_variables))
    assert elements_equal.all()


# todo: add tests involving zeros, infinities, etc.

# todo: do we do checks that parameters stay the same in all periods, as promised in the identify_parameters() method?
