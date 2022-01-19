from dateutil.relativedelta import relativedelta
from collections import defaultdict
from collections import Counter
from statistics import mean
import sys
import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

CCW = ['WEAPONS OFFENSE - CONCEALED', 'WEAPONS - CARRYING A CONCEALED WEAPON (CCW)', 'WEAPONS - FIREARM IN AUTOMOBILE (CCW)']

PANDEMIC_START = datetime.date(2020, 3, 13)


def analysis_start(months=12):
    return PANDEMIC_START + relativedelta(months= - 1 * months)

def analysis_end(months=12):
    return PANDEMIC_START + relativedelta(months=months)


def extract_date(timestamp):
    year, month, day = [int(_) for _ in timestamp[:10].split('/')]
    return datetime.date(year, month, day)

def quarter_from_timestamp(incident_timestamp):
    year_month = incident_timestamp[:7]
    year, month = year_month.split('/')
    month = int(month) - 1
    quarter = int(month / 3) + 1
    return year[2:] + 'Q' + str(quarter)

def month_from_timestamp(incident_timestamp):
    year_month = incident_timestamp[:7]
    year, month = year_month.split('/')
    return int(month)

def is_during_pandemic(timestamp):
    return extract_date(timestamp) >= datetime.date(2020, 3, 12)

def is_during_analysis(timestamp):
    return extract_date(timestamp) > analysis_start() and extract_date(timestamp) < analysis_end()


def load_scout_car_areas():
    scout_car_areas = gpd.read_file('Sources/DPD_Scout_Car_Areas')
    scout_car_areas['Area'] = scout_car_areas.apply(
        lambda x: str(x['Area']), axis=1)
    scout_car_areas['name'] = scout_car_areas.Area
    return scout_car_areas

def load_zips():
    zips = gpd.read_file('Sources/Detroit_Zip_Codes.geojson')
    zips['name'] = zips.ZIPCODE
    return zips

def load_precincts():
    return gpd.read_file('Sources/DPD_Precincts.geojson')

def load_snf():
    snf = gpd.read_file('Sources/SNF.geojson')
    snf['name'] = snf.Proj_NAME
    return snf

def get_processed_rms():
    rms = gpd.read_file('Sources/RMS_Crime_Incidents.geojson')
    # remove excess spaces and identify ccw charges
    rms.charge_description = rms.charge_description.apply(lambda x: x.strip())
    # add a columns indicating whether the charge is ccw or not
    rms['is_ccw'] = rms.charge_description.apply(lambda x: x in CCW)
    rms['incident_timestamp'] = rms.incident_timestamp.apply(lambda x: x.replace('-', '/'))
    # add a quarter column and filter to Jan 2017 - September 2021
    rms['quarter'] = rms.incident_timestamp.apply(lambda x: quarter_from_timestamp(x))
    rms['month'] = rms.incident_timestamp.apply(lambda x: month_from_timestamp(x))

    rms['scout_car_area'] = rms['scout_car_area'].apply(
        lambda x: str(int(x)) if x.isnumeric() else x)
    rms['precinct'] = rms.apply(
        lambda x: str(int(x["precinct"])).zfill(2) 
        if type(x['precinct']) == 'str' 
        and x["precinct"].isnumeric() else x["precinct"], axis=1)
    rms['pandemic'] = rms.incident_timestamp.apply(is_during_pandemic)
    return rms


def get_ccw_only_df(ccw_df=None):
    if ccw_df is None:
       ccw_df = get_processed_rms()

    # ignore it when someone is charged with the same crime multiple
    # times in one arrest
    ccw_df = ccw_df.drop_duplicates(subset=['crime_id', 'is_ccw'])

    # now, we filter out only crimes charged in isolation
    # so that ccw-only charges emerge
    num_charges_per_incident = Counter(ccw_df.crime_id).items()
    singles = [k for k, v in num_charges_per_incident if v == 1]
    singles_df = ccw_df[ccw_df.crime_id.isin(singles)]

    # ccw_only_df consists of only ccw arrests in isolation
    ccw_only_df = singles_df[singles_df.is_ccw]
    return ccw_only_df



def filter_to_during_analysis(ccw_only_df, num_months=20):
    df = ccw_only_df
    analysis_start = PANDEMIC_START + relativedelta(months=-num_months)
    analysis_end = PANDEMIC_START + relativedelta(months=+num_months)
    def is_during_analysis(timestamp):
        ts = extract_date(timestamp)
        return  ts < analysis_end and ts > analysis_start
    df['during_analysis'] = df.incident_timestamp.apply(is_during_analysis)
    df = df[df.during_analysis]
    return df



def custom_vmax(field, analysis):
    if field == 'scout_car_area' and analysis == 'Percent':
        return 10
    return False


def get_color(yoy, pname, field, analysis):
    vmax = custom_vmax(field, analysis)
    if vmax:
        return "black" if yoy.get(pname, 0) < vmax * .8 else 'white'
    return "black"


def get_name(yoy, pname, analysis_field, analysis_type):
    if analysis_type == 'Absolute' or analysis_type == 'Total number of':
        if analysis_field == 'scout_car_area':
            return str(round(yoy.get(pname, 0)))
        if analysis_field == 'zip_code':
            return str(int(pname)) + ':\n' + str(round(yoy.get(pname, 0)))
        if analysis_field == 'SNF':
            return pname + ': ' + str(round(yoy.get(pname, 0)))
        return str(int(pname)) + ': ' + str(round(yoy.get(pname, 0)))
    if analysis_type == 'Percent':
        pct_increase = str(round((yoy.get(pname, 1) - 1) * 100)) + '%'
        if analysis_field == 'scout_car_area':
            return pct_increase
        if analysis_field == 'zip_code':
            return str(int(pname)) + ':\n' + pct_increase
        if analysis_field == 'SNF':
            return pname + ': ' + pct_increase
        return pname + ': ' + pct_increase


def get_yoy(ccws, analysis_type):
    yoy = {}

    if analysis_type == 'Percent':
        for field_val, count in ccws[True].items():
            yoy[field_val] = count  / ccws[False].get(field_val, 1)
            
    elif analysis_type == 'Absolute' or analysis_type == 'Total number of':
        for field_val, count in ccws[True].items():
            if analysis_type == 'Absolute':
                yoy[field_val] = count - ccws[False][field_val]
            elif analysis_type == 'Total number of':
                yoy[field_val] = count
    return yoy

def compare_pandemic(df, field):
    ccws = {True: Counter(), False: Counter()}
    for row in df.iterrows():
        row = row[1]
        val = row[field]
        ccws[row.pandemic][val] += 1
    return ccws






