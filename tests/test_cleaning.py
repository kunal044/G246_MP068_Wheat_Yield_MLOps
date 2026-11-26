# Import testing and data processing libraries
import pytest
import pandas as pd

# Import the project cleaning function
from src.wheat_yield.clean import clean_data


# Create synthetic data for testing
@pytest.fixture
def sample_data():

    return pd.DataFrame({
        "record_id": ["R001", "R001", "R002"],
        "rainfall_mm": [700, 700, 850],
        "fertilizer_kg": [80, 80, 95],
        "sowing_delay_days": [10, 10, 15],
        "irrigation_count": [3, 3, 4],
        "soil_ph": [6.8, 6.8, 7.2],
        "farm_acres": [999, 999, 5.5],
        "seed_variety": ["HD-2967", "HD-2967", "PBW-343"],
        "tehsil": [
            "Tehsil North",
            "Tehsil North",
            "Tehsil South"
        ],
        "yield_from_mandi_receipt": [20, 20, 25],
        "yield_qtl_per_acre": [200, 200, 25]
    })

# Test whether duplicate rows are removed
def test_duplicate_removal(sample_data):

    cleaned = clean_data(sample_data)

    assert len(cleaned) == 2
    assert cleaned.duplicated().sum() == 0

    # Test whether the leaking column is removed
def test_leakage_removal(sample_data):

    cleaned = clean_data(sample_data)

    assert "yield_from_mandi_receipt" not in cleaned.columns

    # Test whether invalid farm sizes become missing values
def test_sentinel_handling(sample_data):

    cleaned = clean_data(sample_data)

    assert (cleaned["farm_acres"] == 999).sum() == 0

    assert cleaned["farm_acres"].isna().sum() == 1

    # Test whether incorrectly multiplied yields are corrected
def test_target_correction(sample_data):

    cleaned = clean_data(sample_data)

    assert cleaned["yield_qtl_per_acre"].tolist() == [
        20.0, 25.0
    ]

    # Verify that cleaning does not modify the original dataframe
def test_original_data_unchanged(sample_data):

    original = sample_data.copy(deep=True)

    clean_data(sample_data)

    pd.testing.assert_frame_equal(
        sample_data,
        original
    )