"""Tests for eeg_autoclean._windowing.apply_persistence_filter."""

import numpy as np

from eeg_autoclean._windowing import apply_persistence_filter


def test_persistence_n_1_is_a_no_op():
    flagged = np.array([[True, False, True, False, True]])
    result = apply_persistence_filter(flagged, persistence_n=1)
    assert np.array_equal(result, flagged)


def test_isolated_single_window_flags_are_dropped_at_n2():
    flagged = np.array([[True, False, True, False, True]])
    result = apply_persistence_filter(flagged, persistence_n=2)
    assert not result.any()


def test_run_of_two_or_more_consecutive_flags_survives_at_n2():
    flagged = np.array([[False, True, True, False, True, True, True]])
    result = apply_persistence_filter(flagged, persistence_n=2)
    expected = np.array([[False, True, True, False, True, True, True]])
    assert np.array_equal(result, expected)


def test_run_shorter_than_n_is_dropped_even_if_others_survive():
    # A 2-window run and a 3-window run in the same channel, persistence_n=3:
    # only the 3-window run should survive.
    flagged = np.array([[True, True, False, True, True, True]])
    result = apply_persistence_filter(flagged, persistence_n=3)
    expected = np.array([[False, False, False, True, True, True]])
    assert np.array_equal(result, expected)


def test_each_channel_filtered_independently():
    flagged = np.array(
        [
            [True, False, True, False],  # all isolated -> all dropped at n=2
            [True, True, False, False],  # one run of 2 -> survives at n=2
        ]
    )
    result = apply_persistence_filter(flagged, persistence_n=2)
    expected = np.array(
        [
            [False, False, False, False],
            [True, True, False, False],
        ]
    )
    assert np.array_equal(result, expected)


def test_run_touching_the_end_of_the_recording_is_still_counted():
    flagged = np.array([[False, False, True, True]])
    result = apply_persistence_filter(flagged, persistence_n=2)
    assert np.array_equal(result, flagged)


def test_no_flags_at_all_stays_empty():
    flagged = np.zeros((3, 10), dtype=bool)
    result = apply_persistence_filter(flagged, persistence_n=2)
    assert not result.any()
