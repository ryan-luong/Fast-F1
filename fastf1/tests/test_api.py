import copy
import datetime

import numpy as np
import pandas as pd
import pytest

import fastf1._api
from fastf1 import Cache


def test_timing_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        lap_data, stream_data = fastf1._api.timing_data(session.api_path)

        # ########## verify lap data
        assert (isinstance(lap_data, pd.DataFrame)
                and isinstance(stream_data, pd.DataFrame))
        assert len(lap_data['Driver'].unique()) == 20  # 20 drivers
        assert len(lap_data.columns) == 18
        assert (lap_data.dtypes == [
            'timedelta64[ns]', 'object', 'timedelta64[ns]', 'int64', 'int64',
            'timedelta64[ns]', 'timedelta64[ns]', 'timedelta64[ns]',
            'timedelta64[ns]', 'timedelta64[ns]', 'timedelta64[ns]',
            'timedelta64[ns]', 'timedelta64[ns]', 'float64', 'float64',
            'float64', 'float64', 'bool']).all()

        # these columns should not contain any NA values
        assert not lap_data.isna().loc[:, ('Time', 'Driver', 'NumberOfLaps',
                                           'NumberOfPitStops')].any().any()
        # double .any(): 1st within column, 2nd column results combined

        # these columns need to contain same number of NA values (outlap)
        count1 = lap_data.isna().loc[:, 'Sector1Time'].sum()
        count2 = lap_data.isna().loc[:, 'Sector1SessionTime'].sum()
        assert count1 == count2
        assert count1 > 50

        # laps with no sector1time need to have a pit out time
        mask = lap_data.isna().loc[:, 'Sector1Time']
        assert not lap_data.loc[mask, 'PitOutTime'].isna().any()

        # sum of sector times needs to be equal to lap time
        mask = (
            ~lap_data.isna().loc[:, ('Sector1Time', 'LapTime')]
        ).all(axis=1)
        sums = lap_data.loc[mask, ('Sector1Time',
                                   'Sector2Time',
                                   'Sector3Time')].sum(axis=1)
        assert np.allclose(
            sums.to_numpy().astype(float),
            lap_data.loc[mask, 'LapTime'].to_numpy().astype(float)
        )

        # ########## verify stream data
        # columns are Time, Driver, Position, GapToLeader, IntervalToPositionAhead
        assert len(stream_data.columns) == 5
        assert (stream_data.dtypes == ['timedelta64[ns]', 'object', 'int64',
                                       'float64', 'float64']).all()
        assert not stream_data.loc[:, ('Time', 'Driver', 'Position')]\
            .isna().any().any()


def test_timing_app_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.timing_app_data(session.api_path)

        # ########## verify lap data
        assert isinstance(data, pd.DataFrame)
        assert len(data.columns) == 13
        assert (data.dtypes == [
            'float64', 'object', 'timedelta64[ns]', 'int64', 'float64',
            'object', 'object', 'object', 'timedelta64[ns]', 'float64',
            'object', 'float64', 'object']).all()


def test_car_data(caplog):
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.car_data(session.api_path)

        assert "failed to decode" not in caplog.text
        assert isinstance(data, dict)
        assert len(data) == 36  # 20 drivers and some problem with the raw data
        assert list(data.values())[0].shape == (27897, 9)  # dataframe shape
        assert (list(data.values())[0].dtypes == [
            'timedelta64[ns]', 'datetime64[ns]', 'int64', 'int64', 'int64',
            'int64', 'bool', 'int64', 'object']).all()


def test_position_data(caplog):
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.position_data(session.api_path)

        assert "failed to decode" not in caplog.text
        assert isinstance(data, dict)
        assert len(data) == 20  # 20 drivers
        assert list(data.values())[0].shape == (26840, 7)  # dataframe shape
        assert (list(data.values())[0].dtypes == [
            'timedelta64[ns]', 'datetime64[ns]', 'object',
            'int64', 'int64', 'int64', 'object']).all()


def test_track_status_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.track_status_data(session.api_path)

        # ########## verify lap data
        assert isinstance(data, dict)
        assert len(data.keys()) == 3
        dtypes = [datetime.timedelta, str, str]
        for col, dtype in zip(data.values(), dtypes):
            assert isinstance(col[0], dtype)
            assert len(col) == 7


def test_session_status_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.session_status_data(session.api_path)

        # ########## verify lap data
        assert isinstance(data, dict)
        assert len(data.keys()) == 2
        dtypes = [datetime.timedelta, str]
        for col, dtype in zip(data.values(), dtypes):
            assert isinstance(col[0], dtype)
            assert len(col) == 5


def test_weather_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2020, 5, "FP2")
        data = fastf1._api.weather_data(session.api_path)

        # ########## verify lap data
        assert isinstance(data, dict)
        assert len(data.keys()) == 8
        dtypes = [
            datetime.timedelta, float, float, float, bool, float, int, float
        ]
        for col, dtype in zip(data.values(), dtypes):
            assert isinstance(col[0], dtype)
            assert len(col) == 100


def test_lap_count_data():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2021, 1, "R")
        data = fastf1._api.lap_count(session.api_path)

        # ########## verify lap data
        assert isinstance(data, dict)
        assert len(data.keys()) == 3
        dtypes = [datetime.timedelta, int, int]
        for col, dtype in zip(data.values(), dtypes):
            assert isinstance(col[0], dtype)
            assert len(col) == 57


def test_driver_list():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2023, 1, "FP1")
        data = fastf1._api.driver_info(session.api_path)

        # ########## verify driver data
        assert isinstance(data, dict)
        assert len(data.keys()) == 20  # 20 drivers in this GP
        dtypes = {'RacingNumber': str, 'BroadcastName': str, 'FullName': str,
                  'Tla': str, 'Line': int, 'TeamName': str, 'TeamColour': str,
                  'FirstName': str, 'LastName': str, 'Reference': str,
                  'HeadshotUrl': str, 'CountryCode': str}
        for driver in data.values():
            assert len(driver.keys()) == 12  # 12 fields in each driver's info.
            for key, val in driver.items():
                assert isinstance(val, dtypes[key])

# ########## special test cases ##########

def test_driver_list_contains_support_race(caplog):
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2023, 11, "FP1")
        data = fastf1._api.driver_info(session.api_path)
        assert len(caplog.record_tuples) == 2
        _, _, warn_message = caplog.record_tuples[1]
        assert warn_message.startswith("Skipping delayed declaration of driver")

        # API data contains drivers from F3 qualifying session
        # beginning half an hour after end of FP1
        session = fastf1.get_session(2025, "Monza", "FP1")
        session.load(laps=False, telemetry=False, weather=False)
        assert len(session.drivers) == 20
        # check F1 driver data is not overwritten
        assert session.results.loc["1", "Abbreviation"] == "VER"


def test_deleted_laps_not_marked_personal_best():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        # see issue #165
        session = fastf1.get_session(2022, 'Spain', 'Q')
        laps_data, _ = fastf1._api.timing_data(session.api_path)
        nor = laps_data[laps_data['Driver'] == '4']  # get NOR laps

        # second to last lap was deleted, ensure it is not marked personal best
        assert (nor['IsPersonalBest']
                == [False, True, False, False, False, False,
                    False, True, False, False, False, False]).all()


def test_personal_best_q_session_handled_individually():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2023, 'Canada', 'Q')
        laps_data, _ = fastf1._api.timing_data(session.api_path)
        ver = laps_data[laps_data['Driver'] == '1']  # get VER laps

        # Over all three Quali sessions, a total of 9 laps should be marked
        # as personal best. Quali was rainy with laps in Q2 and Q3 being slower.
        # If Quali session are not handled correctly individually, those laps are
        # not marked as personal best. (see issue #403)
        assert sum(ver['IsPersonalBest']) == 9


def test_lap_time_of_previous_lap_received_again_ignored():
    # requires clean parsing
    with Cache.disabled(disable_http_cache=False, disable_func_cache=True):
        session = fastf1.get_session(2023, 'Canada', 'Q')
        laps_data, _ = fastf1._api.timing_data(session.api_path)

        # When the session was resumed after the red flags in Q1 and Q3, the
        # API sent the lap time, sector 3 time and finish line speed of the
        # previous lap again for these in-laps. These values must not be
        # assigned to the in-laps. (see issue #404)
        for drv, lap_n in (('1', 25), ('11', 2), ('2', 3), ('4', 24),
                           ('55', 23)):
            lap = laps_data[(laps_data['Driver'] == drv)
                            & (laps_data['NumberOfLaps'] == lap_n)].iloc[0]
            assert pd.isna(lap['LapTime'])
            assert pd.isna(lap['Sector3Time'])
            assert pd.isna(lap['SpeedFL'])
            # an in-lap cannot end before the car enters the pits
            assert lap['Time'] > lap['PitInTime']


def _resent():
    # lap time, sector 3 time and finish line speed of lap 1
    return {'LastLapTime': {'Value': '1:30.000'},
            'Sectors': {'2': {'Value': '30.000'}},
            'Speeds': {'FL': {'Value': '280'}}}


def _laps_from_driver_raw(*messages):
    # parse synthetic timing data of one driver, lap 1 starts at 00:00:02
    driver_raw = [('00:00:01.000', {'InPit': False}),
                  ('00:00:02.000', {'NumberOfLaps': 1}), *messages]
    driver_raw_before = copy.deepcopy(driver_raw)
    laps, _ = fastf1._api._laps_data_driver(driver_raw,
                                            fastf1._api.EMPTY_LAPS, '1')
    # the raw data must not be modified by the parser
    assert driver_raw == driver_raw_before
    return pd.DataFrame(laps).set_index('NumberOfLaps')


def test_lap_time_of_previous_lap_received_again_ignored_synthetic():
    # (see issue #404)
    lap_1_end = ('00:01:32.000', {'NumberOfLaps': 2, **_resent()})
    lap_2_end = ('00:04:00.000', {'NumberOfLaps': 3})

    # Values of lap 1 that are received again during lap 2 are ignored.
    # Other values in the same message belong to lap 2.
    resent = _resent()
    resent['Speeds']['I1'] = {'Value': '250'}
    laps = _laps_from_driver_raw(lap_1_end,
                                 # a list instead of a dict must not crash
                                 ('00:02:30.000', {'LastLapTime': []}),
                                 ('00:03:00.000', resent), lap_2_end)
    assert pd.isna(laps.loc[2, 'LapTime'])
    assert pd.isna(laps.loc[2, 'Sector3Time'])
    assert pd.isna(laps.loc[2, 'SpeedFL'])
    assert laps.loc[2, 'SpeedI1'] == 250

    # An empty lap time of lap 2 is not calculated from the sector times,
    # because the sector 3 time is the one of lap 1.
    laps = _laps_from_driver_raw(
        lap_1_end, ('00:02:02.000', {'Sectors': {'0': {'Value': '29.000'}}}),
        ('00:02:50.000', {'LastLapTime': {'Value': ''},
                          'Sectors': {'1': {'Value': '30.000'},
                                      '2': {'Value': '30.000'}}}),
        ('00:03:00.000', _resent()), lap_2_end)
    assert pd.isna(laps.loc[2, 'LapTime'])

    # A different lap time of lap 2 that was received before is kept.
    laps = _laps_from_driver_raw(
        lap_1_end, ('00:03:00.000', {'LastLapTime': {'Value': '1:28.000'}}),
        ('00:03:00.100', _resent()), lap_2_end)
    assert laps.loc[2, 'LapTime'] == pd.Timedelta(seconds=88)


def test_lap_time_equal_to_previous_lap_kept_synthetic():
    # Values equal to those of the previous lap are kept if they are received
    # with the start of a new lap or up to five seconds after it.
    # (see issue #404)
    laps = _laps_from_driver_raw(
        ('00:01:32.000', {'NumberOfLaps': 2,
                          'LastLapTime': {'Value': '1:30.000'}}),
        ('00:01:33.000', _resent()),
        ('00:03:02.000', {'NumberOfLaps': 3, **_resent()}))
    assert laps.loc[1, 'Sector3Time'] == pd.Timedelta(seconds=30)
    assert laps.loc[2, 'LapTime'] == pd.Timedelta(seconds=90)

    # A lap time equal to an earlier lap, but not to the previous one, is
    # kept, also if it is received before the start of the next lap.
    laps = _laps_from_driver_raw(
        ('00:01:32.000', {'NumberOfLaps': 2, **_resent()}),
        ('00:03:03.000', {'NumberOfLaps': 3,
                          'LastLapTime': {'Value': '1:31.000'}}),
        ('00:04:33.000', {'LastLapTime': {'Value': '1:30.000'}}),
        ('00:04:34.500', {'NumberOfLaps': 4}))
    assert laps.loc[3, 'LapTime'] == pd.Timedelta(seconds=90)
