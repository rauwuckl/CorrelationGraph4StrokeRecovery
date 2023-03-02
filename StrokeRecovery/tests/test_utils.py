
from StrokeRecovery import utils

import pandas as pd

import unittest


class TestUtils(unittest.TestCase):


    def test_load_schema(self):

        schema = utils.get_config()
        print(schema)
    
    def test_rename_dict(self):
        rename_dict = utils.get_category_rename_dict()

    def test_patient_summary(self):

        patients = pd.read_excel('data/ds2a_patients_v1.6.xlsx', index_col=[0])

        sum_stats=  utils.SummaryStatistics(patients)

        mean_sd_example = sum_stats._mean_sd(sum_stats.table['duration_of_stay'])

        counts_example = sum_stats._percentages(sum_stats.table['Geschlecht_SN'], rename={1: 'ga', 2: 'halbtax'}) 

        percentiles_example = sum_stats._percentiles(sum_stats.table['duration_of_stay'])


        overview_table = sum_stats.get_overview_table(
            [
                ('duration_of_stay', 'duration', 'meanSd'),
                ('Geschlecht_SN', 'Geschlecht', 'percentages', dict(rename={1: 'ga', 2: 'halbtax'}) ),
                ('duration_of_stay', 'duration percent', 'percentiles'),
            ]
        )