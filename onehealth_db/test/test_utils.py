import pytest
import json
from pathlib import Path
from onehealth_db import utils


def get_files(dir_path: Path, name_phrase: str) -> list[Path]:
    """
    Get all files in a directory that contain the name_phrase in their name.
    """
    return [
        file
        for file in dir_path.iterdir()
        if file.is_file() and name_phrase in file.name
    ]


def test_is_valid_settings():
    settings = {"adjust_longitude": True}
    assert utils.is_valid_settings(settings) is True
    settings = {"adjust_longitude": "error"}
    assert utils.is_valid_settings(settings) is False

    settings = {"convert_kelvin_to_celsius": True}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_kelvin_to_celsius": "error"}
    assert utils.is_valid_settings(settings) is False

    settings = {"convert_m_to_mm_precipitation": True}
    assert utils.is_valid_settings(settings) is True
    settings = {"convert_m_to_mm_precipitation": "error"}
    assert utils.is_valid_settings(settings) is False

    settings = {"downsample_grid": 1}
    assert utils.is_valid_settings(settings) is True
    settings = {"downsample_grid": 1.5}
    assert utils.is_valid_settings(settings) is True
    settings = {"downsample_grid": "error"}
    assert utils.is_valid_settings(settings) is False

    settings = {"truncate_year": 2025}
    assert utils.is_valid_settings(settings) is True
    settings = {"truncate_year": 1.5}
    assert utils.is_valid_settings(settings) is False
    settings = {"truncate_year": "error"}
    assert utils.is_valid_settings(settings) is False

    settings = {"rename": {"t2m": "temperature"}}
    assert utils.is_valid_settings(settings) is True
    settings = {"rename": {"t2m": 1}}
    assert utils.is_valid_settings(settings) is False
    settings = {"rename": {"t2m": "temperature", "error": 1}}
    assert utils.is_valid_settings(settings) is False
    settings = {"rename": "error"}
    assert utils.is_valid_settings(settings) is False


def test_update_new_settings_empty():
    updated = utils._update_new_settings({"test": "test"}, {})
    assert updated is False

    with pytest.raises(ValueError):
        utils._update_new_settings({}, {"test": "test"})


def test_update_new_settings_not_updated():
    # invalid key
    with pytest.warns(UserWarning):
        updated = utils._update_new_settings(
            {"adjust_longitude": True}, {"test": "test"}
        )
    assert updated is False

    # invalid structure
    with pytest.warns(UserWarning):
        updated = utils._update_new_settings(
            {"adjust_longitude": True}, {"adjust_longitude": 1}
        )
    assert updated is False
    with pytest.warns(UserWarning):
        updated = utils._update_new_settings(
            {"rename": {"t2m": "temperature"}}, {"rename": {"t2m": 1}}
        )
    assert updated is False

    # same value
    updated = utils._update_new_settings(
        {"adjust_longitude": False}, {"adjust_longitude": False}
    )
    assert updated is False
    updated = utils._update_new_settings(
        {"rename": {"t2m": "temperature"}}, {"rename": {"t2m": "temperature"}}
    )
    assert updated is False


def test_update_new_settings_updated():
    settings = {"adjust_longitude": False}
    updated = utils._update_new_settings(settings, {"adjust_longitude": True})
    assert updated is True
    assert settings.get("adjust_longitude") is True

    settings = {"rename": {"t2m": "temperature"}}
    updated = utils._update_new_settings(
        settings, {"rename": {"t2m": "temp", "tcc": "cloud_cover"}}
    )
    assert updated is True
    assert settings.get("rename") == {"t2m": "temp", "tcc": "cloud_cover"}


def test_save_settings_to_file(tmpdir):
    settings = {"adjust_longitude": False}

    # none dir path
    utils.save_settings_to_file(settings)
    saved_files = get_files(Path.cwd(), "updated_settings_")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False
    saved_files[0].unlink()  # remove the file

    # valid dir path
    directory = Path(tmpdir.mkdir("test"))
    utils.save_settings_to_file(settings, directory)
    saved_files = get_files(directory, "updated_settings_")
    assert len(saved_files) == 1
    with open(saved_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False

    # invalid dir path
    file_path = Path(__file__).absolute()
    with pytest.raises(ValueError):
        utils.save_settings_to_file(settings, file_path)


def test_get_settings_default():
    settings = utils.get_settings()
    assert settings.get("adjust_longitude") is True

    settings = utils.get_settings("default")
    assert settings.get("adjust_longitude") is True


def test_get_settings_file(tmp_path):
    setting_path = tmp_path / "settings.json"

    # invalid cases
    # not existing file
    with pytest.warns(UserWarning):
        settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is True

    # empty file
    open(setting_path, "w", newline="", encoding="utf-8").close()
    with pytest.warns(UserWarning):
        settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is True

    # invalid json file
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        f.write("test")
    with pytest.warns(UserWarning):
        settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is True

    # invalid json file against the schema
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"test": "test"}, f)
    with pytest.warns(UserWarning):
        settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is True

    # valid json file
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"adjust_longitude": False}, f)
    settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is False


def test_get_settings_new_settings(tmp_path, tmpdir):
    new_settings = {"adjust_longitude": False}

    # update default settings
    settings = utils.get_settings(
        new_settings=new_settings, save_updated_settings=False
    )
    assert settings.get("adjust_longitude") is False

    # update settings from file
    setting_path = tmp_path / "settings.json"
    with open(setting_path, "w", newline="", encoding="utf-8") as f:
        json.dump({"adjust_longitude": True}, f)
    settings = utils.get_settings(setting_path)
    assert settings.get("adjust_longitude") is True
    settings = utils.get_settings(
        setting_path, new_settings=new_settings, save_updated_settings=False
    )
    assert settings.get("adjust_longitude") is False

    # update settings from file with invalid new settings
    new_settings = {"test": "test"}
    with pytest.warns(UserWarning):
        settings = utils.get_settings(setting_path, new_settings=new_settings)
    assert settings.get("adjust_longitude") is True

    # update settings from file, save updated settings
    new_settings = {"adjust_longitude": False}
    directory = Path(tmpdir.mkdir("test"))
    settings = utils.get_settings(
        setting_path,
        new_settings=new_settings,
        updated_setting_dir=directory,
        save_updated_settings=True,
    )
    assert settings.get("adjust_longitude") is False
    created_files = get_files(directory, "updated_settings_")
    assert len(created_files) == 1
    with open(created_files[0], "r", encoding="utf-8") as f:
        updated_settings = json.load(f)
    assert updated_settings.get("adjust_longitude") is False
