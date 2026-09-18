from pathlib import Path
import numpy as np
import pandas as pd
from esik_assessment import observed_scenario_candidates

ROOT=Path(__file__).resolve().parents[1]


def test_demo_company_never_presents_imputed_sales_growth_as_observed_scenario():
    rows=pd.read_csv(ROOT/'outputs/esik_scored_test_companies.csv').set_index('COMPANY_ID')
    shap=pd.read_csv(ROOT/'outputs/esik_shap_local_values.csv')
    company=rows.loc['ESIK-05817']
    selected=observed_scenario_candidates(shap.loc[shap.COMPANY_ID.eq('ESIK-05817')],company)
    assert pd.isna(company.Attr21)
    assert selected.VARIABLE.tolist()==['Attr39','Attr35','Attr24']
    assert all(np.isfinite(float(company[f])) for f in selected.VARIABLE)


def test_missing_flag_and_raw_observation_are_both_required():
    shap=pd.DataFrame({'VARIABLE':['Attr1','Attr2','Attr3','Attr4'],
        'ABS_SHAP_VALUE':[9.,8.,7.,6.], 'WAS_MISSING':['True','False',False,False]})
    row=pd.Series({'Attr1':12.,'Attr2':np.nan,'Attr3':float('inf'),'Attr4':.4})
    assert observed_scenario_candidates(shap,row).VARIABLE.tolist()==['Attr4']
    assert observed_scenario_candidates(shap,row*float('nan')).empty
