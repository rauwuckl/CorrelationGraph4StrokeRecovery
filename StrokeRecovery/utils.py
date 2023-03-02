import json
import importlib.resources
from turtle import pos
import numpy as np
import pandas as pd
from .register_functions import class_register, register


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

_loaded_config = None
def get_config():
    global _loaded_config
    if _loaded_config is None:
        with importlib.resources.open_text("StrokeRecovery.resources", "config.json") as file:
            data = json.load(file)

        _loaded_config = data

    return _loaded_config

#####################################################################################
### The following functions help with changing labels as they appear in the paper ###
#####################################################################################
def _built_rename_dict(name_of_dict):
    rename_raw = get_config()[name_of_dict]
    
    out = dict()
    for propername, list_of_aliases in rename_raw.items():
        for alias in list_of_aliases:
            if alias in out.keys():
                prev = out[alias]
                other = propername
                raise ValueError(">{}< not uniquely defined in {}, it is set to both >{}< and >{}< ".format(alias, name_of_dict, prev, other))
            
            out[alias] = propername
    return out

_all_rename_dicts = dict()
def _get_rename_dict(name_of_dict, linebreak_level=False):
    global _all_rename_dicts
    if name_of_dict not in _all_rename_dicts:
        this = _built_rename_dict(name_of_dict)
        _all_rename_dicts[name_of_dict] = this

    this_dict = _all_rename_dicts[name_of_dict]
        
    if linebreak_level == 1:
        return {tmpname: propername.replace("\\1\\", "\n").replace("\\2\\", " ") for tmpname, propername in this_dict.items()}
    elif linebreak_level == 2:
        return {tmpname: propername.replace("\\1\\", "\n").replace("\\2\\", "\n")  for tmpname, propername in this_dict.items()}
    elif linebreak_level == 0:
        return {tmpname: propername.replace("\\1\\", " ").replace("\\2\\", " ")  for tmpname, propername in this_dict.items()}
    else:
        raise ValueError("linebreak_level={} should be one of 0,1, 2".format(linebreak_level))


def get_category_rename_dict(linebreaks=0):
    return _get_rename_dict('proper_category_names', linebreak_level=linebreaks)
def get_loc_rename_dict(linebreaks=0):
    return _get_rename_dict('proper_loc_names', linebreak_level=linebreaks)
def get_diag_rename_dict(linebreaks=0):
    return _get_rename_dict('proper_diag_names', linebreak_level=linebreaks)

def categories():
    return list(get_config()['categories'].keys())


#######################

@class_register
class SummaryStatistics:
    """
        Given a DataFrame, produce an overview table with summary statistics (e.g. mean, percentiles...) for a set of columns.
        The kind of summary statiscs available are defined as methods that are decorated with @register(<name of summary>)
    """
    def __init__(self, table, tab_characters="      ", empty_cell_character="."):
        """
        Args:
            table (pd.DataFrame): the table of which to compute summary statistics
        """
        self.table = table
        self.tab_characters = tab_characters
        self.empty_cell_character = empty_cell_character


    _mean_sd_template = "{mean:.1f} ({std:.3f})"
    @register('meanSd')
    def _mean_sd(self, column):
        assert not np.any(pd.isnull(column))

        mean = column.mean()
        std = column.std()

        formated = self._mean_sd_template.format(mean=mean, std=std)
        return formated


    _percentage_template = "{n:d} ({val:.1f}%)"
    @register('percentages')
    def _percentages(self, column, rename=None):
        counts = column.value_counts().sort_index()

        percentages = counts/counts.sum()

        collector = list()

        for rname, fraction in percentages.items():
            name = rname
            if fraction > 0.005:
                n = counts[name]
                if rename is not None:
                    name = rename[name]

                formated_precentage = self._percentage_template.format(n=n, val = fraction*100)
                this = (name, formated_precentage)
                collector.append(this)

        return collector

    _percentiles_template = "{m50:.1f} ({m25:.1f} - {m75:.1f})"
    @register('percentiles')
    def _percentiles(self, column):

        missing_fraction = pd.isnull(column).astype(float).mean()

        m50 = column.quantile(0.5)
        m25 = column.quantile(0.25)
        m75 = column.quantile(0.75)

        median = column.median()
        assert m50==median

        text =  self._percentiles_template.format(m50=m50, m25=m25, m75=m75) 

        if missing_fraction >0:
            text += ", {:.1f}% missing values".format(missing_fraction*100)
        return text



    _count_template = "N={}"
    @register('count')
    def _count(self, column):
        return self._count_template.format(len(column))

    def _get_function(self, fname):
        if not fname in self.registered_functions:
            raise ValueError('Unkown {}'.format(fname))
        else:
            return self.registered_functions[fname]

    def get_overview_table(self, columns_and_commands):
        """make an overview pd.Series as by instructions

        Args:
            columns_and_commands (List[Tuple3[str, str, str] | str]): each tuple in the list contains: name of the column, name of the row in the summary, name of the aggregate function, (optinal: dictionary with extra keywords for the function)
                if a command is a single string, an empty header row is inserted
        """
        collector = list()

        for position in columns_and_commands:
            if type(position) == str: # insert an empty header column
                collector.append((position, '.'))
                continue 
            elif len(position) == 3:
                colname, summarycolname, funcname = position
                extra_args = {}
            elif len(position) == 4:
                colname, summarycolname, funcname, extra_args = position
            else:
                raise ValueError("Unexpected Tuple: {}".format(position))

            func = self._get_function(funcname)
            this_column = self.table[colname]

            result_this = func(this_column, **extra_args)
            # the result is either a single value or a list of tuples with subcolumns
            if type(result_this) == list:
                collector.append((summarycolname, self.empty_cell_character))
                for name, val in result_this:
                    collector.append((self.tab_characters + str(name), val))
            else:
                collector.append((summarycolname, result_this))


        idxs, vals = zip(*collector)
        return pd.Series(vals, index=idxs)




