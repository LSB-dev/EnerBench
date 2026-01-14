import pandas as pd
import os
import random

# Use only 1000 loads
use_short_data = False

dir_path = r"D:\EnerBench\RawData\raw_load_data"
res_path = r"D:\EnerBench\RawData"
os.makedirs(res_path, exist_ok=True)

files = ["hlt_profiles_tabsep.csv", "load_profiles_tabsep.csv", "master_data_tabsep.csv"]

data_dict = {}

for fname in files:
    file_path = os.path.join(dir_path, fname)
    dat = pd.read_csv(file_path, sep="\t")
    # Shorten data
    if use_short_data:
        dat = dat.iloc[:min(len(dat), 1000),:]

    data_dict[fname] = dat

# Read parquets
master_data = data_dict[files[2]]
load_data = data_dict[files[1]]

unique_sectors = master_data.loc[:, "Sector_group_id"].dropna().unique()
unique_sector_labels = {}
for sect in unique_sectors:
    unique_sector_labels[sect] = (master_data[master_data["Sector_group_id"]==sect].Sector_group.iloc[0])


selected_master_data = master_data.copy()

selected_load_data = load_data[load_data.id.isin(selected_master_data.Id)]
selected_load_data.index = selected_load_data.id
selected_load_data.drop("id", axis=1, inplace=True)

# Exploratory

df = selected_load_data.T

time_index = pd.date_range(
    start="2016-01-01 00:00:00",
    end="2017-01-01 00:00:00",
    freq="15min"
)

df.index = time_index

summary = df.describe()
summary.loc["missing",:] = df.isna().sum(axis=0)

# Add metadata
summary_t = summary.T
master = selected_master_data.set_index("Id")

summary_with_meta = summary_t.join(master)
summary_with_meta.loc[:,"Id"] = summary_with_meta.index

summary_with_meta.to_csv(rf"{res_path}/summary.csv")


# Merge with weather data
import requests
from datetime import datetime
from meteostat import Point, Hourly

# Approximate coordinates
import numpy as np
import pgeocode

#from meteostat import Stations

_nomi_de = pgeocode.Nominatim("de")

def plz2_to_centroid_and_state(plz2: str):
    """
    Returns:
      (lat, lon, state_name)
    computed from all 5-digit DE postal codes starting with the given 2-digit prefix.
    No manual mapping table needed.
    """
    plz2 = str(plz2).zfill(2)

    # pgeocode dataset (DataFrame)
    codes = _nomi_de._data.copy()

    # postal_code is sometimes numeric/float -> cast safely to string
    pc = codes["postal_code"].astype(str).str.replace(r"\.0$", "", regex=True)

    subset = codes[pc.str.startswith(plz2)].copy()

    # keep rows with valid coordinates
    subset = subset.dropna(subset=["latitude", "longitude"])
    if subset.empty:
        raise ValueError(f"No matches / no coordinates for PLZ prefix {plz2}")

    lat = float(subset["latitude"].mean())
    lon = float(subset["longitude"].mean())

    # state name (mode = most frequent)
    state_col = "state_name" if "state_name" in subset.columns else None
    state = None
    if state_col:
        m = subset[state_col].dropna()
        state = str(m.mode().iloc[0]) if not m.empty else None

    return lat, lon, state

lat_long = {}
for id, zip_prefix in zip(selected_master_data.Id, selected_master_data.Zip_code):
    lat_long[id] = plz2_to_centroid_and_state(plz2=str(zip_prefix))

# Get weather data
def get_weather_2016_meteostat(lat, lon):
    start = datetime(2016, 1, 1)
    end   = datetime(2016, 12, 31, 23, 59)
    p = Point(lat, lon)
    weather_h = Hourly(p, start, end).fetch()
    weather_h.index = pd.to_datetime(weather_h.index)
    return weather_h

def merge_weather_15min(df_load_15min, weather_hourly):
    # Upsample hourly → 15min and forward-fill (common for weather features)
    weather_15 = weather_hourly.resample("15min").ffill()

    # Align to load index and merge
    weather_15 = weather_15.reindex(df_load_15min.index, method="ffill")
    return df_load_15min.join(weather_15, how="left")

load_datasets = {}
for load_var in df.columns:
    weather_dat = get_weather_2016_meteostat(lat_long[load_var][0], lat_long[load_var][1])
    merged_dat = merge_weather_15min(df.loc[:, [load_var]], weather_dat)
    load_datasets[load_var] = merged_dat.rename(columns={merged_dat.columns[0]: "load"})

# Add calendar variables
import holidays

def add_basic_calendar_features(df):
    out = df.copy()
    idx = out.index

    out["dow"] = idx.dayofweek          # 0=Mon .. 6=Sun
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["hour"] = idx.hour
    out["minute"] = idx.minute
    out["month"] = idx.month
    out["is_workday"] = (out["dow"] < 5).astype(int)
    return out

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    # --- daily cycle (minute-of-day) ---
    minute_of_day = idx.hour * 60 + idx.minute
    out["sin_day"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    out["cos_day"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

    # --- weekly cycle (day-of-week) optional but often useful ---
    dow = idx.dayofweek  # 0..6
    out["sin_week"] = np.sin(2 * np.pi * dow / 7)
    out["cos_week"] = np.cos(2 * np.pi * dow / 7)

    # --- yearly cycle (day-of-year; handles leap years) ---
    doy = idx.dayofyear  # 1..365/366
    days_in_year = 366 if idx.is_leap_year.any() else 365
    out["sin_year"] = np.sin(2 * np.pi * doy / days_in_year)
    out["cos_year"] = np.cos(2 * np.pi * doy / days_in_year)

    return out

def add_holiday_features_by_plz2(df, state, years=None, time_col=None):
    """
    Adds:
      - is_holiday (0/1)
      - holiday_name (string, optional)
    based on German state approximated from PLZ2.
    """
    out = df.copy()
    idx = out.index if time_col is None else pd.to_datetime(out[time_col])

    if years is None:
        years = sorted(set(idx.year))

    # Handle exceptions in state category
    if state == "Lower Saxony":
        state = "Niedersachsen"
    if state == "Bavaria":
        state = "Bayern"
    if state == "Land Berlin":
        state = "Berlin"

    de_holidays = holidays.country_holidays("DE", subdiv=state, years=years)

    dates = idx.normalize()
    out["is_holiday"] = dates.isin(de_holidays).astype(int)

    # Optional: holiday name (string). Comment out if you don’t need it.
    out["holiday_name"] = dates.map(lambda d: de_holidays.get(d.date()) if d.date() in de_holidays else None)

    return out

for load_var in load_datasets.keys():
    basic_calendar_extension = add_basic_calendar_features(load_datasets[load_var])
    cyclical_time_extension = add_cyclical_time_features(basic_calendar_extension)
    calendar_extended_dat = add_holiday_features_by_plz2(cyclical_time_extension, lat_long[load_var][2])
    load_datasets[load_var] = calendar_extended_dat

# Add suffix
all_dat = []
for i in range(len(load_datasets)):
    all_dat.append(load_datasets[i].add_suffix(f"_{str(master_data.Id[i])}"))

all_dat = pd.concat(all_dat, axis=1)

# Save selected datasets and an exploratory analysis
all_dat.to_csv(rf"{res_path}/all_loads.csv")

## Create middle sized sample dataset

# Select 15 vars randomly
rng = np.random.default_rng(345)
nums = random.sample(range(1000), 15)

cols = [
    c for c in all_dat.columns
    if any(c.endswith(f"_{i}") for i in nums)
]

selected = all_dat[cols]
selected.to_csv(rf"{res_path}/15_loads.csv")


# Select random sample within sector
sectors = summary_with_meta['Sector_group_id'].unique()

selected_ids = []
for sector in sectors:
    sector_dat = summary_with_meta[summary_with_meta['Sector_group_id'] == sector]
    if len(sector_dat) > 0:
        if len(sector_dat) >= 10:
            rng = np.random.default_rng(75)
            nums = random.sample(range(len(sector_dat)), 10)
            selected_ids = selected_ids + list(sector_dat.iloc[nums,:].Id)
        else:
            print("Not enough datapoints in sector" + sector_dat.Sector_group.iloc[0])

cols = [
    c for c in all_dat.columns
    if any(c.endswith(f"_{i}") for i in selected_ids)
]

selected = all_dat[cols]
selected.to_csv(rf"{res_path}/section_sample.csv")
