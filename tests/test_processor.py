import sys
sys.path.append("../")
import importlib
import pandas as pd


PROCESSOR = getattr(importlib.import_module("src.processor"), "Processor")

def test_extract_titles() -> None:
    """Test extract_titles method."""
    data = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina"]})
    processor = PROCESSOR(data)
    result = processor._Processor__extract_titles(processor.data.Name.str).values.tolist()
    assert result == ["Mr", "Miss"]

def test_replace_rare_titles() -> None:
    """Test replace_rare_titles method."""
    data = pd.DataFrame({"Title": ["Mr", "Miss", "Lady", "Capt", "Dr"]})
    processor = PROCESSOR(data)
    result = processor._Processor__replace_rare_titles(processor.data.Title).values.tolist()
    assert result == ["Mr", "Miss", "Rare", "Rare", "Rare"]

def test_replace_miss_variants() -> None:
    """Test replace_mlle_ms_variants method."""
    data = pd.DataFrame({"Title": ["Mr", "Miss", "Mlle", "Ms"]})
    processor = PROCESSOR(data)
    result = processor._Processor__replace_miss_variants(processor.data.Title).values.tolist()
    assert result == ["Mr", "Miss", "Miss", "Miss"]

def test_replace_mrs_variants() -> None:
    """Test replace_mrs_variants method."""
    data = pd.DataFrame({"Title": ["Mr", "Mrs", "Mme"]})
    processor = PROCESSOR(data)
    result = processor._Processor__replace_mrs_variants(processor.data.Title).values.tolist()
    assert result == ["Mr", "Mrs", "Mrs"]

def test_title_mapping() -> None:
    """Test title_mapping method."""
    data = pd.DataFrame({"Title": ["Mr", "Miss", "Mrs", "Master", "Rare"]})
    processor = PROCESSOR(data)
    result = processor._Processor__title_mapping(processor.data.Title).values.tolist()
    assert result == [1, 2, 3, 4, 5]

def test_compute_family_size() -> None:
    """Test compute_family_size method."""
    data = pd.DataFrame({"SibSp": [1, 2], "Parch": [3, 4]})
    processor = PROCESSOR(data)
    processor._compute_family_size()
    result = processor.data.FamilySize.values.tolist()
    assert result == [5, 7]

def test_compute_is_alone() -> None:
    """Test compute_is_alone method."""
    data = pd.DataFrame({"FamilySize": [0, 1, 2]})
    processor = PROCESSOR(data)
    processor._compute_is_alone()
    result = processor.data.IsAlone.values.tolist()
    assert result == [1, 0, 0]

def test_compute_title() -> None:
    """Test compute_title method."""
    data = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina"]})
    processor = PROCESSOR(data)
    processor._compute_title()
    result = processor.data.Title.values.tolist()
    assert result == [1, 2]

def test_compute_age_band() -> None:
    """Test compute_age_band method."""
    data = pd.DataFrame({"Age": [0, 5, 12, 18, 24, 35, 60, 80]})
    processor = PROCESSOR(data)
    processor._compute_age_band()
    result = processor.data.AgeBand.values.tolist()
    assert result == [1, 3, 3, 4, 5, 5, 5, 5]

def test_compute_fare_band() -> None:
    """Test compute_fare_band method."""
    data = pd.DataFrame({"Fare": [0, 7.91, 14.454, 31, 120, 512.329]})
    processor = PROCESSOR(data)
    processor._compute_fare_band()
    result = processor.data.FareBand.values.tolist()
    assert result == [0, 0, 1, 2, 3, 3]

def test_compute_sex() -> None:
    data = pd.DataFrame({"Sex": ["male", "female"]})
    processor = PROCESSOR(data)
    processor._compute_sex()
    result = processor.data.Sex.values.tolist()
    assert result == [1, 0]