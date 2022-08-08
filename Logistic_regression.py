## Python 3.9.13
## Building a logistic regression model
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import datetime, date
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

## Loadind dataset
file = 'test_task_dataset_created_20210611.xlsx'
xl = pd.ExcelFile(file)
df1 = xl.parse('Sheet1')
print("Dataset loaded successfully")

## Calculating intervals
df1.loc[df1['crdeal_deallife__first_loan_open_date'] == "00.01.1900  0:00:00", 'crdeal_deallife__first_loan_open_date'] = np.nan
df1.loc[df1['crdeal_deallife__last_loan_open_date__donor_mfo'] == "00.01.1900  0:00:00", 'crdeal_deallife__last_loan_open_date__donor_mfo'] = np.nan
df1.loc[df1['crdeal_deallife__last_loan_open_date__donor_bnk'] == "00.01.1900  0:00:00", 'crdeal_deallife__last_loan_open_date__donor_bnk'] = np.nan
df1.loc[df1['crdeal_deallife__first_loan_open_date__donor_bnk'] == "00.01.1900  0:00:00", 'crdeal_deallife__first_loan_open_date__donor_bnk'] = np.nan
df1.loc[df1['crdeal_deallife__first_loan_open_date__donor_mfo'] == "00.01.1900  0:00:00", 'crdeal_deallife__first_loan_open_date__donor_mfo'] = np.nan
df1.loc[df1['crdeal_deallife__last_loan_open_date'] == "00.01.1900  0:00:00", 'crdeal_deallife__last_loan_open_date'] = np.nan
df1.loc[(df1['crdeal_deallife__first_loan_open_date']).astype(str) == "00:00:00", 'crdeal_deallife__first_loan_open_date'] = np.nan
df1.loc[(df1['crdeal_deallife__last_loan_open_date__donor_mfo']).astype(str) == "00:00:00", 'crdeal_deallife__last_loan_open_date__donor_mfo'] = np.nan
df1.loc[(df1['crdeal_deallife__last_loan_open_date__donor_bnk']).astype(str) == "00:00:00", 'crdeal_deallife__last_loan_open_date__donor_bnk'] = np.nan
df1.loc[(df1['crdeal_deallife__first_loan_open_date__donor_bnk']).astype(str) == "00:00:00", 'crdeal_deallife__first_loan_open_date__donor_bnk'] = np.nan
df1.loc[(df1['crdeal_deallife__first_loan_open_date__donor_mfo']).astype(str) == "00:00:00", 'crdeal_deallife__first_loan_open_date__donor_mfo'] = np.nan
df1.loc[(df1['crdeal_deallife__last_loan_open_date']).astype(str) == "00:00:00", 'crdeal_deallife__last_loan_open_date'] = np.nan

## Droping NaN values
df1=df1.dropna()
print("Strings with NULL-values droped successfully")

## Converting intervals into days and cjnverting flags
for i in df1.index:
    df1.at[i,"Interval_4_6"] =(datetime.date(df1.at[i,"crdeal_deallife__last_loan_open_date__donor_mfo"]) - datetime.date(df1.at[i,"crdeal_deallife__last_loan_open_date__donor_bnk"])).days
    df1.at[i,"Interval_4_5"] =(datetime.date(df1.at[i,"crdeal_deallife__last_loan_open_date__donor_mfo"]) - datetime.date(df1.at[i,"crdeal_deallife__first_loan_open_date__donor_bnk"])).days
    df1.at[i,"Interval_4_2"] =(datetime.date(df1.at[i,"crdeal_deallife__last_loan_open_date__donor_mfo"]) - datetime.date(df1.at[i,"crdeal_deallife__last_loan_open_date"])).days
    if df1.at[i,"target_flag"]=="good":
        df1.at[i,"target_flag"]=int("1")
    if df1.at[i,"target_flag"]=="bad":
        df1.at[i,"target_flag"]=int("0")
    if df1.at[i,"target_flag"]=="not determined":
        df1.at[i,"target_flag"]=""

## Removing parameters with invalid sigma    
##  0
df1.drop(["crdeal_deallife__first_loan_open_date","crdeal_deallife__last_loan_open_date__donor_mfo", "crdeal_deallife__last_loan_open_date__donor_bnk", "crdeal_deallife__first_loan_open_date__donor_bnk", "crdeal_deallife__first_loan_open_date__donor_mfo", "crdeal_deallife__last_loan_open_date", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_7_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_360_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_7_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_45_days", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_7_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_exst__last_active_gt_360_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_7_days", "crdeal_deallife__loans_cnt__state_sold", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_45_days", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_60_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_60_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_30_days", "crdeal_deallife__initial_amount_min__donor_bnk", "crdeal_deallife__initial_amount_min__donor_mfo", "crdeal_deallife__initial_amount_min__state_clsd", "crdeal_deallife__overdue_debt_sum", "crdeal_deallife__overdue_debt_sum__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_1", "crdeal_deallife__loans_cnt__current_dpd_gt_1__state_clsd", "crdeal_deallife__historical_max_overdue_debt", "crdeal_deallife__historical_max_overdue_debt__donor_mfo", "crdeal_deallife__historical_max_overdue_debt__clsd_state", "crdeal_deallife__last_loan_status__donor_bnk", "credres__credit_req_cnt__org_bnk__last_1_days", "Interval_4_6", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_30_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_120_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_7_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_gt_360_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_gt_360_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_7_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_90_days", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_120_days", "crdeal_deallife__loans_cnt__state_sold__opened_lteq_180_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_discounted", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_90_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_gt_360_days", "crdeal_deallife__initial_amount_sum", "crdeal_deallife__initial_amount_min", "crdeal_deallife__initial_amount_min__consumer_loan", "crdeal_deallife__plan_debt_mean", "crdeal_deallife__overdue_debt_max", "crdeal_deallife__overdue_debt_sum__donor_bnk", "crdeal_deallife__overdue_debt_max__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_5__state_clsd", "crdeal_deallife__historical_max_overdue_debt__donor_bnk", "crdeal_deallife__current_max_dpd", "crdeal_deallife__current_max_dpd__donor_mfo", "credres__non_credit_req_cnt__any_org__any_reqdatetime", "credres__credit_req_cnt__org_bnk__any_reqdatetime", "credres__credit_req_cnt__org_bnk__last_3_5_days", "credres__credit_req_cnt__org_bnk__last_7_days", "Interval_4_5", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_7_days", "crdeal_deallife__loans_cnt__state_exst__last_active_lteq_7_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_120_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_120_days", "crdeal_deallife__initial_amount_max__donor_mfo", "crdeal_deallife__overdue_debt_mean", "crdeal_deallife__overdue_debt_max__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_5", "crdeal_deallife__loans_cnt__current_dpd_gt_10__state_clsd", "crdeal_deallife__historical_max_dpd__donor_mfo", "crdeal_deallife__current_max_dpd__donor_bnk", "credres__credit_req_cnt__org_mfo__last_1_days", "credres__credit_req_cnt__org_bnk__last_10_days", "Interval_4_2", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__state_discounted__opened_lteq_180_days", "crdeal_deallife__overdue_debt_mean__donor_bnk", "crdeal_deallife__overdue_debt_mean__donor_mfo", "crdeal_deallife__overdue_debt_sum__state_exst_or_sold_or_bought", "crdeal_deallife__loans_cnt__current_dpd_gt_10", "crdeal_deallife__loans_cnt__current_dpd_gt_1__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_20__state_clsd", "crdeal_deallife__loans_cnt__dldonor_bnk", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_45_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_90_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_360_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_45_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_45_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_90_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_30_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_120_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__state_sold__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__state_discounted__opened_gt_360_days", "crdeal_deallife__loans_cnt__state_discounted__last_active_lteq_180_days", "crdeal_deallife__initial_amount_max", "crdeal_deallife__initial_amount_mean", "crdeal_deallife__initial_amount_sum__donor_bnk", "crdeal_deallife__initial_amount_sum__donor_mfo", "crdeal_deallife__initial_amount_min__state_exst", "crdeal_deallife__initial_amount_mean__state_exst", "crdeal_deallife__initial_amount_sum__state_clsd", "crdeal_deallife__initial_amount_max__state_clsd", "crdeal_deallife__initial_amount_mean__state_clsd", "crdeal_deallife__initial_amount_max__consumer_loan", "crdeal_deallife__initial_amount_mean__consumer_loan", "crdeal_deallife__plan_debt_max__donor_mfo", "crdeal_deallife__overdue_debt_max__state_exst_or_sold_or_bought", "crdeal_deallife__loans_cnt__current_dpd_gt_20", "crdeal_deallife__loans_cnt__current_dpd_gt_60", "crdeal_deallife__loans_cnt__current_dpd_gt_1__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_5__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_20__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_30__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_60__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_5__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_90__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_1__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_5__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_20__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_30__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_60__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_30__state_clsd", "credres__credit_req_cnt__any_org__any_reqdatetime", "credres__credit_req_cnt__any_org__any_reqdatetime", "credres__credit_req_cnt__org_mfo__last_3_5_days", "credres__credit_req_cnt__org_mfo__last_60_days", "credres__credit_req_cnt__org_mfo__more_than_360_days", "credres__credit_req_cnt__org_bnk__last_13_days", "credres__credit_req_cnt__org_bnk__last_30_days", "credres__credit_req_cnt__org_bnk__more_than_90_days", "credres__credit_req_cnt__org_bnk__last_360_days", "credres__credit_req_cnt__org_bnk__more_than_360_days", "crdeal_deallife__loans_cnt__dlrolesub_zaemshik__curr_uah", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_gt_360_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_360_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_gt_360_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_60_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_60_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_120_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_45_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_30_days", "crdeal_deallife__initial_amount_mean__donor_bnk", "crdeal_deallife__initial_amount_sum__consumer_loan", "crdeal_deallife__plan_debt_sum", "crdeal_deallife__plan_debt_max", "crdeal_deallife__plan_debt_sum__donor_bnk", "crdeal_deallife__plan_debt_max__donor_bnk", "crdeal_deallife__plan_debt_sum__donor_mfo", "crdeal_deallife__plan_debt_max__state_exst", "crdeal_deallife__plan_debt_mean__state_exst", "crdeal_deallife__overdue_debt_mean__state_exst_or_sold_or_bought", "crdeal_deallife__loans_cnt__current_dpd_gt_30", "crdeal_deallife__loans_cnt__current_dpd_gt_90", "crdeal_deallife__loans_cnt__current_dpd_gt_10__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_90__donor_bnk", "crdeal_deallife__loans_cnt__current_dpd_gt_10__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_20__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_60__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_90__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_60__state_clsd", "crdeal_deallife__historical_max_overdue_debt__exst_state", "crdeal_deallife__first_loan_status", "crdeal_deallife__first_loan_status__donor_bnk", "crdeal_deallife__last_loan_status", "crdeal_deallife__last_loan_status__donor_mfo", "credres__credit_req_cnt__org_mfo__any_reqdatetime", "credres__credit_req_cnt__org_mfo__last_10_days", "credres__credit_req_cnt__org_mfo__last_13_days", "credres__credit_req_cnt__org_mfo__last_20_days", "credres__credit_req_cnt__org_mfo__last_30_days", "credres__credit_req_cnt__org_mfo__last_360_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_180_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_45_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_gt_360_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_60_days", "crdeal_deallife__plan_debt_sum__state_exst", "crdeal_deallife__loans_cnt__current_dpd_gt_30__donor_mfo", "crdeal_deallife__loans_cnt__current_dpd_gt_90__state_clsd", "credres__credit_req_cnt__org_bnk__last_20_days", "crdeal_deallife__loans_cnt__dldonor_bnk__last_active_gt_360_days", "crdeal_deallife__loans_cnt__dldonor_mfo", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_60_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_90_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_90_days", "crdeal_deallife__loans_cnt__state_exst__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_90_days", "crdeal_deallife__loans_cnt__state_closed__opened_gt_360_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_sold__last_active_gt_360_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_120_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_120_days", "crdeal_deallife__loans_cnt__dldonor_mfo__opened_lteq_180_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_120_days", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_120_days", "crdeal_deallife__loans_cnt__state_exst__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_120_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_180_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_120_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__state_closed__last_active_gt_360_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_60_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_360_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_120_days", "credres__credit_req_cnt__org_mfo__more_than_90_days", "crdeal_deallife__loans_cnt__dldonor_bnk__opened_lteq_60_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__dldonor_mfo__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__state_exst", "crdeal_deallife__loans_cnt__state_exst__opened_lteq_360_days", "crdeal_deallife__loans_cnt__state_exst__opened_gt_360_days", "crdeal_deallife__loans_cnt__state_exst__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_45_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_120_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_30_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_45_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_60_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_360_days", "crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_180_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__state_exst__last_active_lteq_90_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_120_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_180_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__state_closed__opened_lteq_360_days", "crdeal_deallife__loans_cnt__state_closed__last_active_lteq_360_days", "crdeal_deallife__loans_cnt__consumer_loan", "crdeal_deallife__loans_cnt__consumer_loan__opened_gt_360_days", "crdeal_deallife__loans_cnt__consumer_loan__last_active_gt_360_days", "crdeal_deallife__loans_cnt__state_prolonged_restructured__opened_lteq_180_days"], axis=1, inplace = True)

## Sequential removal of parameters that have little effect on the result
##  1
df1.drop(["crdeal_deallife__loans_cnt__state_exst__last_active_lteq_360_days"], axis=1, inplace = True)
##  2
df1.drop(["crdeal_deallife__loans_cnt__state_exst__last_active_lteq_180_days"], axis=1, inplace = True)
##  3
df1.drop(["crdeal_deallife__historical_max_dpd"], axis=1, inplace = True)
##  4
df1.drop(["crdeal_deallife__historical_max_dpd__donor_bnk"], axis=1, inplace = True)
##  5
df1.drop(["crdeal_deallife__loans_cnt__dlrolesub_zaemshik__curr_not_uah"], axis=1, inplace = True)

## This step achieves the highest accuracy
##  6
df1.drop(["crdeal_deallife__initial_amount_sum__state_exst"], axis=1, inplace = True)
####  7
##df1.drop(["crdeal_deallife__loans_cnt__state_exst__last_active_lteq_120_days"], axis=1, inplace = True)
####  8
##df1.drop(["crdeal_deallife__loans_cnt__current_dpd_gt_10__state_exst"], axis=1, inplace = True)
####  9
##df1.drop(["crdeal_deallife__loans_cnt__state_sold__opened_gt_360_days"], axis=1, inplace = True)
####  10
##df1.drop(["crdeal_deallife__loans_cnt__state_sold__opened_lteq_360_days"], axis=1, inplace = True)
####  11
##df1.drop(["crdeal_deallife__first_loan_status__donor_mfo"], axis=1, inplace = True)
####  12
##df1.drop(["crdeal_deallife__loans_cnt__consumer_loan__opened_lteq_90_days"], axis=1, inplace = True)
####  13
##df1.drop(["crdeal_deallife__loans_cnt__state_closed__last_active_lteq_30_days"], axis=1, inplace = True)
####  14
##df1.drop(["credres__credit_req_cnt__org_bnk__last_60_days"], axis=1, inplace = True)
####  15
##df1.drop(["crdeal_deallife__loans_cnt__state_exst__opened_lteq_180_days"], axis=1, inplace = True)
####  16
##df1.drop(["credres__credit_req_cnt__org_mfo__last_7_days"], axis=1, inplace = True)
####  17
##df1.drop(["crdeal_deallife__plan_debt_mean__donor_bnk"], axis=1, inplace = True)
####  18
##df1.drop(["crdeal_deallife__initial_amount_max__state_exst"], axis=1, inplace = True)
####  19
##df1.drop(["crdeal_deallife__loans_cnt__consumer_loan__last_active_lteq_90_days"], axis=1, inplace = True)
####  20
##df1.drop(["crdeal_deallife__loans_cnt__state_closed"], axis=1, inplace = True)
####  21
##df1.drop(["crdeal_deallife__loans_cnt__state_closed"], axis=1, inplace = True)

print("Dataset optimized successfully")

## Separation of records for which prediction is required
df_learn=df1[df1["target_flag"]!=""]
df_c=df1[df1["target_flag"]==""]
df_learn=df_learn.astype('float64')
df_check=df_c.drop("target_flag", axis=1)
df_check=df_check.astype('float64')
print("Dataset splitted successfully")

## Preparing data for building a model
X = df_learn.loc[:, df_learn.columns != 'target_flag']
y = df_learn.loc[:, df_learn.columns == 'target_flag']

## Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

## Oversampling data
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['target_flag'])
print("Dataset oversampled successfully")

## Analizing parameters
logit_model=sm.Logit(os_data_y['target_flag'],os_data_X)
result=logit_model.fit(maxiter=1024)
print(result.summary2())

## Model Training
logreg1 = LogisticRegression(max_iter=1024, solver ="liblinear")
logreg1.fit(X_train, y_train.values.ravel())
y_check1 = logreg1.predict(X_train)
print('Accuracy of logistic regression by liblinear classifier on train set: {:.4f}'.format(logreg1.score(X_train, y_train)))
y_pred1 = logreg1.predict(X_test)
print('Accuracy of logistic regression by liblinear classifier on test set: {:.4f}'.format(logreg1.score(X_test, y_test)))

## Analizing model quality
confusion_matrix = confusion_matrix(y_test, y_pred1)
print("Confusion_matrix:\n",confusion_matrix)
print("Classification_report:\n",classification_report(y_test, y_pred1))

## Preparing parameters for output
check_pred=logreg1.predict(df_check).astype('int')
default=logreg1.predict_proba(df_check)
default_score=[]
for i in default:
    default_score.append(round(i[0]*100,2))
params=logreg1.feature_names_in_
coef=logreg1.coef_.ravel()
parameters={"Name":params,"Coeff":coef}
parameters_out=pd.DataFrame(parameters)
df_check['Probability of Default, %'] = default_score
df_check['target_flag'] = check_pred.astype('int')
df_check['target_flag']=df_check['target_flag'].replace([0,1],["bad","good"])

## Outputting parameters to a file
writer = pd.ExcelWriter('Model.xlsx', engine='xlsxwriter')
parameters_out.to_excel(writer, 'Parameters')
df_check.to_excel(writer, 'Predictions')
writer.save()
print("Model outed successfully")

## Printing ROC
logit_roc_auc = roc_auc_score(y_test, logreg1.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg1.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()











