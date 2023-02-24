from ccdecomp import CostModel

def test_title():
    cp = CostModel(title='Title', equation='x1 x2 + x3 x4')
    assert cp.title == 'Title'

def test_setEquationString():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._equation == 'x1 x2 + x3 x4'

    cp = CostModel('x1 *x2 + x3 x4')
    assert cp._equation == 'x1 x2 + x3 x4'

    cp = CostModel('  x1 *x2   + x3 x4  ')
    assert cp._equation == 'x1 x2 + x3 x4'

def test_setEquationDefineTermsUsingSpaces():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._cost_component_terms == ['x1 x2', 'x3 x4']

    cp = CostModel('x1*x2 + x3 x4')
    assert cp._cost_component_terms == ['x1 x2', 'x3 x4']

    cp = CostModel('x1* x2 + x3 x4')
    assert cp._cost_component_terms == ['x1 x2', 'x3 x4']

def test_setEquationCountTerms():
    cp = CostModel('x1 x2 + x3 x4')
    assert cp._n_components == 2