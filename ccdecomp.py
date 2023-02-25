import re
import numpy as np
import pandas as pd
from utils import unique, setdiff
from typing import List

def logmean(C1,C2):    
    import numpy as np
    return (C2-C1) / (np.log(C2) - np.log(C1))

class CostModel:
    """Cost model object for performing cost change decomposition.

    A cost model object is a holder for information about a cost change decomposition problem.  It consists of an mathematical equation that represents the cost model, data to populate the model, a variety of other calculated quantities, and meta data about the problem.
    """    
    

    def __init__(self, equation: str, title: str = 'Cost model'):
        """Cost model object for performing cost change decomposition.

        A cost model object is a holder for information about a cost change decomposition problem.  It consists of an mathematical equation that represents the cost model, data to populate the model, a variety of other calculated quantities, and meta data about the problem.

        Parameters
        ----------
        equation : str
            A cost equation and the species the cost model.
         
            To be parsed correctly the equation should follow these rules:
            
            (1) No equals sign: Do not include an "=" sign. Specify only the
            right side of the equation.
            
            (2) Addition: Use "+" to add terms together, e.g.
                "x1 + x2"
                "x1+x2"
            
            (3) Multiplication: One can either
                (i)  use "*" to multiply factors together, or
                (ii) write factors next to one another with a space, e.g.
                "x1*x2"
                "x1 * x2"
                "x1 x2"
            
            (4) Variable names: Use variable names that would be accepted by
            python, e.g.
                "x"
                "logEfficiency"
                "speed_cubed"
            
                are okay, while "total$" or "#_of_supports" are not.
        title : str, optional
            A title for the cost model, by default 'Cost model'
        """        
        
        self.title = title
        self._equation = self._regularize_equation(equation)
        self._cost_component_exprs = self._parse_cost_components()
        self._n_components = len(self._cost_component_exprs)

        symbols = self._gather_symbols()
        self._symbols = symbols
        self._n_symbols = len(symbols)

        # Until told otherwise, assume all symbols are variables, not parameters
        self._variables = symbols
        self._n_variables = len(symbols)
        self._parameters = []
        self._n_parameters = 0

        # Give cost components default names
        self._cost_component_names = ['C'+str(i+1) for i in range(4)]

        # Compute matrix of dependencies of cost components on symbols
        self._dependency_matrix = self._construct_dependency_matrix()

        # Specify styles for display methods
        self._styles = [ dict(selector='caption', props=[('text-align', 'left'),('font-weight', 'bold'), ('text-decoration','underline')]) ]


    def _regularize_equation(self, equation):
        equation = (equation
                    .strip()
                    .replace('*', ' ')
                    .replace('+', ' + '))
        equation = re.sub(' +',' ',equation)  # replace multiple spaces with just one
        return equation

    def _parse_cost_components(self):
        cost_component_terms = self._equation.split('+')
        cost_component_terms = [term.strip() for term in cost_component_terms]
        return cost_component_terms

    def _gather_symbols(self):
        symbols = []
        for term in self._cost_component_exprs:
            new_symbols = term.split(' ')
            symbols = symbols + new_symbols
        symbols = unique(symbols)
        return symbols
    
    def _construct_dependency_matrix(self):
        dependency_matrix = np.zeros((self._n_components, self._n_symbols))
        for i, term in enumerate(self._cost_component_exprs):
            factors = term.split(' ')
            dependency_matrix[i,:] = np.isin(self._symbols, factors)
        return dependency_matrix
    
    def __str__(self):
        S = '\n'.join((
            'Title:                  ' + self.title,
            'Equation:               ' + self._equation,
            'Cost comp. names:       ' + str(self._cost_component_names),
            'Cost comp. expressions: ' + str(self._cost_component_exprs),
            'Symbols:                ' + str(self._symbols),
            'Variables:              ' + str(self._variables),
            'Parameters:             ' + str(self._parameters),
            'Num. cost components:   ' + str(self._n_components),
            'Num. symbols:           ' + str(self._n_symbols),
            'Num. variables:         ' + str(self._n_variables),
            'Num. parameters:        ' + str(self._n_parameters),
            'Dependency matrix: \n' + str(self._dependency_matrix)
            ))
        return S

    def name_cost_components(self, names: List[str]):        
        """Override default names of cost components.

        The tables that return change decomposition results use these cost components names, so assigning meaningful names can make these results more readable.

        Parameters
        ----------
        names : List[str]
            Names of the additive terms (cost components) of the cost model.
        """
        self._cost_component_names = names

    def identify_parameters(self, parameters: List[str]):
        """Specify which symbols of the cost model represent fixed parameters.

        Specifying the fixed parameters of the model is not required to compute cost change contributions, but has several benefits:
         
             *The cost change contributions of parameters (which are zero since they do not change) are removed from output.
         
             *If data values of parameters were provided previously, then checks are performed that the values of parameters are the same in all periods.
         
             *If data values of parameters were not provided yet, then they can be entered using this function.

        Internally, all symbols (parameters and variables) are handled as variables, with parameters being variables that stay the same in all periods.

        Parameters
        ----------
        parameters : List[str]
            List of parameter names.
        """                
        self._parameters = parameters
        self._n_parameters = len(parameters)
        variables = setdiff(self._variables, parameters)
        self._variables = variables
        self._n_variables = len(variables)


    def bind_data(self, data: pd.DataFrame):
        """Bind data to the variables of the cost model.

        This method is used to provide data values for all symbols of the cost model.  Data is entered as a pandas dataframe, whose columns match the names of symbols used in the cost equation (both variables and parameters) and rows indexed by 'time'.  Upon binding the problem to data, each component of cost is automatically computed from this data for each time period.  The cost components and variable data is recorded in a table, which can be reviewed using the `display_data' method.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame with columns matching variables and parameters used in the cost equation, and rows indexed by 'time'.
        """        
        assert isinstance(data, pd.DataFrame)
        self._data = data.copy()
        self._data.index.name = 'Time period'
        self._n_timeperiods = data.shape[0]
        self._compute_cost_components()


    def _compute_cost_components(self):
        """Uses input data to compute cost components in each period."""

        # Calculate cost components for all time periods
        for i,term in enumerate(self._cost_component_exprs):
            factors = term.split(' ')
            cost_component_value = self._data[factors].product(axis=1)
            cost_component_name = self._cost_component_names[i]
            self._data[cost_component_name] = cost_component_value

        # Calculate total cost
        self._data['Total_cost'] = self._data[self._cost_component_names].sum(axis=1)

        # Calculate cost shares
        for ccname in self._cost_component_names:
            sharename = 'Share_' + ccname
            self._data[sharename] = self._data[ccname] / self._data['Total_cost']


    def cost_change_decomposition(self, time_spans):
        """Compute cost change contributions over one more spans of time.

        This method computes the individual contribution of each variable to change in total cost over one more more spans of time.  The results are recorded in a table that can be accessed by the `display_contributions` method.  Several auxilliary quantities that were used to compute cost change contributions can also be accessed using the `display_change_data' method.

        Parameters
        ----------
        time_spans : list of tuples
            A list of tuples of the form (period1, period2), where period1 and period2 are start and end periods for spans of time over which contributions to cost change should be computed.  These periods should match indices of the data table provided through `bind_data`.
        """     

        self._n_timespans = len(time_spans)

        # Convert time span tuples to strings
        span_strings = []
        for span in time_spans:
            t1 = span[0]
            t2 = span[1]
            span_strings = span_strings + [str(t1) + '-' + str(t2)]

        # Create a table to store representative cost components
        self._representative_costs = pd.DataFrame(index=span_strings, columns=self._cost_component_names)
        self._representative_costs.index.name = 'Time span'

        # Create a table to store variable changes
        self._variable_changes = pd.DataFrame(index=span_strings, columns=self._variables)
        self._variable_changes.index.name = 'Time span'

        # Create a table to store cost change contributions
        col_names = ['Total','Sum_of_changes(vars)'] + self._cost_component_names + self._variables
        self._DeltaCost = pd.DataFrame(index=span_strings, columns=col_names)
        self._DeltaCost.index.name = 'Time span'

        # Compute representative cost components and cost change contributions
        self._DeltaC_matrix_over_time = []
        for span in time_spans:
            t1 = span[0]
            t2 = span[1]
            span_string = str(t1) + '-' + str(t2)

            # Compute total cost change in this time span
            data = self._data
            cost_change = data.loc[t2,'Total_cost'] - data.loc[t1,'Total_cost']
            self._DeltaCost.loc[span_string,'Total'] = cost_change

            # Compute representative value of cost components in this time span
            for ccname in self._cost_component_names:
                C1 = self._data.loc[t1,ccname]
                C2 = self._data.loc[t2,ccname]
                self._representative_costs.loc[span_string, ccname] = logmean(C1,C2)
                self._DeltaCost.loc[span_string,ccname] = C2 - C1

            # Compute log changes to each equation variable during this span
            for vname in self._symbols:
                v1 = self._data.loc[t1,vname]
                v2 = self._data.loc[t2,vname]
                self._variable_changes.loc[span_string, vname] = np.log(v2/v1)
                
            # Compute the contribution of each variable to each cost component
            Cvec          = self._representative_costs.loc[span_string, self._cost_component_names]
            Dlog_var_vec  = self._variable_changes.loc[span_string, self._symbols]
            Cdiag         = np.diag(Cvec)
            Dlog_var_diag = np.diag(Dlog_var_vec)
            D             = np.array(self._dependency_matrix)
            DeltaC_matrix = Cdiag .dot(D) .dot(Dlog_var_diag)
            self._DeltaC_matrix_over_time = self._DeltaC_matrix_over_time + [DeltaC_matrix] #save results from this period

            # Compute the total contribution of each variable to total cost change
            self._DeltaCost.loc[span_string, self._symbols] = DeltaC_matrix.sum(axis=0)

            # Re-compute total cost change from the variable contributions as a check
            self._DeltaCost.loc[span_string, 'Sum_of_changes(vars)'] = self._DeltaCost.loc[span_string, self._symbols].sum()


    pd.set_option('display.precision', 3)

    def display_data(self):
        """Display a table of symbol data in each period."""     
        display(self._data)

    def display_contributions(self):
        """Display a table of cost change contributions from each variable over each time span."""
        display(self._DeltaCost.drop(columns=self._parameters))

    def display_change_data(self):
        """Display representatitive costs over each time space."""
        display(self._representative_costs)
        
    def display_log_changes(self):
        """Display log changes in variables over each time span."""
        display(self._variable_changes.drop(columns=self._parameters))