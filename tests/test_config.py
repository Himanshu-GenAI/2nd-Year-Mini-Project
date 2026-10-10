from datetime import datetime, timedelta

from src import config


def _to_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def test_config_values_load_and_dates_in_order():
    study_start = _to_date(config.STUDY_START)
    study_end = _to_date(config.STUDY_END)
    kbdi_start = _to_date(config.KBDI_SPINUP_START)

    train_start = _to_date(config.TRAIN_START)
    train_end = _to_date(config.TRAIN_END)
    val_start = _to_date(config.VAL_START)
    val_end = _to_date(config.VAL_END)
    test_start = _to_date(config.TEST_START)
    test_end = _to_date(config.TEST_END)

    assert kbdi_start < study_start
    assert study_start <= train_start < train_end < val_start < val_end < test_start < test_end <= study_end


def test_date_gaps_between_splits():
    train_end = _to_date(config.TRAIN_END)
    val_start = _to_date(config.VAL_START)
    val_end = _to_date(config.VAL_END)
    test_start = _to_date(config.TEST_START)

    assert (val_start - train_end).days - 1 == 2
    assert (test_start - val_end).days - 1 == 2


def test_label_windows_never_cross_split():
    train_end = _to_date(config.TRAIN_END)
    val_start = _to_date(config.VAL_START)
    val_end = _to_date(config.VAL_END)
    test_start = _to_date(config.TEST_START)
    horizon = timedelta(days=config.LABEL_HORIZON_DAYS)

    assert train_end + horizon < val_start
    assert val_end + horizon < test_start


def test_test_end_within_study_end():
    test_end = _to_date(config.TEST_END)
    study_end = _to_date(config.STUDY_END)
    horizon = timedelta(days=config.LABEL_HORIZON_DAYS)

    assert test_end + horizon <= study_end


def test_study_and_spinup_durations():
    study_start = _to_date(config.STUDY_START)
    study_end = _to_date(config.STUDY_END)
    kbdi_start = _to_date(config.KBDI_SPINUP_START)

    # Number of days inclusive
    study_days = (study_end - study_start).days + 1
    spinup_days = (study_end - kbdi_start).days + 1

    assert study_days == 4018
    assert spinup_days == 4383


def test_train_start_history_buffer():
    study_start = _to_date(config.STUDY_START)
    train_start = _to_date(config.TRAIN_START)

    assert (train_start - study_start).days >= config.LABEL_PREV_DAYS - 1


def test_grid_candidate_cells():
    lat_cells = (config.LAT_MAX - config.LAT_MIN) / config.GRID_DEG
    lon_cells = (config.LON_MAX - config.LON_MIN) / config.GRID_DEG

    assert lat_cells == 64
    assert lon_cells == 62
