import tempfile
import shutil
from pathlib import Path
import toml
import pytest

from super_segment.app_config import AppConfig, find_project_root

def print_temp_config_tree(temp_dir):
    """
    Print the directory tree and contents of all config files in the temp config dir.
    """
    from pathlib import Path

    def print_file(path, indent="    "):
        print(f"{indent}{path.name}")
        content = path.read_text()
        for line in content.splitlines():
            print(f"{indent}    {line}")

    temp_dir = Path(temp_dir)
    print(f"{temp_dir.name}/")
    for item in sorted(temp_dir.iterdir()):
        if item.is_dir():
            print(f"    {item.name}/")
            for subitem in sorted(item.iterdir()):
                print_file(subitem, indent="        ")
        else:
            print_file(item, indent="    ")

def test_print_temp_config_tree(temp_config_dir):
    print_temp_config_tree(temp_config_dir)

@pytest.fixture
def temp_config_dir():
    # Set up a temporary directory with config files
    temp_dir = tempfile.mkdtemp()
    conf_dir = Path(temp_dir) / "conf"
    conf_dir.mkdir(parents=True)
    # Main config
    main_config = {
        "app": {"title": "Test App"},
        "sub_config": {"files": ["ui.toml", "generate.toml"]},
        "data": {"n_member": 123, "random_seed": 99}
    }
    (conf_dir / "app_config.toml").write_text(toml.dumps(main_config))
    # Sub-config: ui.toml
    ui_config = {"sidebar": {"age": {"min": 18, "max": 99}}}
    (conf_dir / "ui.toml").write_text(toml.dumps(ui_config))
    # Sub-config: generate.toml
    generate_config = {"data": {"n_member": 456, "random_seed": 42, "output_file": "foo.parquet"}}
    (conf_dir / "generate.toml").write_text(toml.dumps(generate_config))
    # Project root marker
    (Path(temp_dir) / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_find_project_root(temp_config_dir):
    # Should find the marker file
    marker = "pyproject.toml"
    root = find_project_root(marker=marker)
    assert root.exists()
    assert (root / marker).exists()


def test_main_config_loading(temp_config_dir, monkeypatch):
    print(f"TEMP DIR: {temp_config_dir}")
    print(f"MARKER EXISTS: {(Path(temp_config_dir) / 'pyproject.toml').exists()}")
    cfg = AppConfig(main_path="app_config.toml", conf_dir="conf", project_root=temp_config_dir)
    assert cfg.get("app", "title") == "Test App"
    assert cfg.get("data", "n_member") == 123

def test_sub_config_loading(temp_config_dir, monkeypatch):
    monkeypatch.chdir(temp_config_dir)
    cfg = AppConfig(main_path="app_config.toml", conf_dir="conf", project_root=temp_config_dir)
    # Should load sub-configs
    assert "ui" in cfg.sub_configs
    assert "generate" in cfg.sub_configs
    assert cfg.get("sidebar", "age", "min", sub_name="ui") == 18
    assert cfg.get("data", "n_member", sub_name="generate") == 456

def test_default_and_missing_keys(temp_config_dir, monkeypatch):
    monkeypatch.chdir(temp_config_dir)
    cfg = AppConfig(main_path="app_config.toml", conf_dir="conf", project_root=temp_config_dir)
    assert cfg.get("nonexistent", "key", default="foo") == "foo"
    assert cfg.get("sidebar", "age", "nonexistent", sub_name="ui", default=0) == 0

def test_as_dict_and_debug_print(temp_config_dir, monkeypatch, capsys):
    monkeypatch.chdir(temp_config_dir)
    cfg = AppConfig(main_path="app_config.toml", conf_dir="conf", project_root=temp_config_dir)
    d = cfg.as_dict()
    assert "app" in d
    assert "ui" in d
    assert "generate" in d
    cfg.debug_print()
    out = capsys.readouterr().out
    assert "Test App" in out

def test_verbose_echo_source(temp_config_dir, monkeypatch, capsys):
    monkeypatch.chdir(temp_config_dir)
    cfg = AppConfig(main_path="app_config.toml", conf_dir="conf", project_root=temp_config_dir)
    # Main config
    val = cfg.get("app", "title", verbose=True)
    out = capsys.readouterr().out
    assert "MAIN config" in out
    assert "Test App" in val
    # Sub-config
    val = cfg.get("data", "n_member", sub_name="generate", verbose=True)
    out = capsys.readouterr().out
    assert "SUB-CONFIG 'generate'" in out
    assert val == 456
    # Not found
    val = cfg.get("foo", "bar", verbose=True, default="baz")
    out = capsys.readouterr().out
    assert "NOT FOUND" in out
    assert val == "baz"

def test_load_sub_config_file_not_found(temp_config_dir, monkeypatch):
    # Remove a sub-config file and check error handling
    monkeypatch.chdir(temp_config_dir)
    (Path(temp_config_dir) / "conf" / "ui.toml").unlink()
    with pytest.raises(FileNotFoundError):
        AppConfig()
