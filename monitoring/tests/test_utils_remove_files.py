import os
import utils

def test_remove_files(tmp_path):
    # Create a temp file
    f = tmp_path / "temp.csv"
    f.write_text("hello")

    # Ensure it exists
    assert f.exists()

    # Remove file
    utils.remove_files(str(f))

    # File should be gone
    assert not f.exists()
