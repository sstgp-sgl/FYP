"""Importers: ASCII/binary PLY, missing RGB fallback, empty clouds, Gaussian
detection, upload caps, reconstruction copy restrictions."""
import numpy as np
import pytest

from src.tour.importers import (
    import_from_reconstruction,
    import_from_upload,
    parse_ply_data,
    sanitize_name,
    seed_demo,
)
from src.tour.repository import (
    InvalidDataError,
    NotFoundError,
    PayloadTooLargeError,
    TourRepository,
)
from tests.tour.conftest import make_config, make_points, write_ascii_ply


def test_parse_ascii_rgb(tmp_path):
    points, colors = make_points(count=120)
    path = write_ascii_ply(tmp_path / "a.ply", points, colors)
    parsed, parsed_colors, missing = parse_ply_data(path.read_bytes(), make_config(), "test")
    assert parsed.shape == (120, 3)
    assert np.array_equal(parsed_colors, colors)
    assert missing is False


def test_parse_binary_rgb(tmp_path):
    points, colors = make_points(count=120)
    path = write_ascii_ply(tmp_path / "b.ply", points, colors, as_binary=True)
    parsed, parsed_colors, missing = parse_ply_data(path.read_bytes(), make_config(), "test")
    assert parsed.shape == (120, 3)
    assert missing is False


def test_missing_rgb_gets_default_gray(tmp_path):
    points, _ = make_points(count=50)
    path = write_ascii_ply(tmp_path / "c.ply", points, colors=None)
    parsed, parsed_colors, missing = parse_ply_data(path.read_bytes(), make_config(), "test")
    assert missing is True
    assert parsed_colors.shape == (50, 3)
    assert (parsed_colors == np.array([180, 180, 180])).all()


def test_empty_ply_rejected(tmp_path):
    path = write_ascii_ply(tmp_path / "empty.ply", np.zeros((0, 3)))
    with pytest.raises(InvalidDataError):
        parse_ply_data(path.read_bytes(), make_config(), "test")


def test_gaussian_fields_rejected(tmp_path):
    points, colors = make_points(count=60)
    path = write_ascii_ply(
        tmp_path / "g.ply", points, colors,
        extra_props=[
            ("f_dc_0", "f4", np.zeros(60, dtype=np.float32)),
            ("opacity", "f4", np.ones(60, dtype=np.float32)),
        ],
    )
    with pytest.raises(InvalidDataError, match="Gaussian"):
        parse_ply_data(path.read_bytes(), make_config(), "test")


def test_nan_coordinates_rejected(tmp_path):
    points, colors = make_points(count=60)
    points[3] = [np.nan, 0.0, 0.0]
    path = write_ascii_ply(tmp_path / "nan.ply", points, colors)
    with pytest.raises(InvalidDataError):
        parse_ply_data(path.read_bytes(), make_config(), "test")


def test_point_cap_enforced(tmp_path):
    points, colors = make_points(count=200)
    path = write_ascii_ply(tmp_path / "big.ply", points, colors)
    config = make_config({"tour": {"max_upload_bytes": 104857600, "max_points": 100, "preview_points": 50}})
    with pytest.raises(PayloadTooLargeError):
        parse_ply_data(path.read_bytes(), config, "test")


def test_byte_cap_on_upload(tmp_path, tour_root):
    points, colors = make_points(count=3000)
    path = write_ascii_ply(tmp_path / "large.ply", points, colors)
    data = path.read_bytes()
    config = make_config({"tour": {"max_upload_bytes": len(data) // 2, "max_points": 2000000, "preview_points": 100}})
    repository = TourRepository(tour_root)
    with pytest.raises(PayloadTooLargeError):
        import_from_upload(repository, data, "large", config)


def test_upload_creates_scene_and_preview(tmp_path, tour_root):
    points, colors = make_points(count=800, seed=4)
    path = write_ascii_ply(tmp_path / "ok.ply", points, colors)
    repository = TourRepository(tour_root)
    scene = import_from_upload(repository, path.read_bytes(), "上传场景", make_config())
    assert scene.revision == 1
    assert scene.source_kind == "real"
    loaded = repository.get_scene(scene.scene_id)
    assert loaded.asset_sha256 == scene.asset_sha256
    metadata = repository.ensure_preview(repository.assets.get(scene.asset_id), make_config())
    assert metadata["total_points"] == 800
    assert metadata["shown_points"] == 800  # below the preview cap
    assert (tour_root / "previews" / scene.asset_sha256 / "points.bin").is_file()
    # uploads must not mutate the scene on re-import (new scene, new asset copy)
    scene2 = import_from_upload(repository, path.read_bytes(), "上传场景2", make_config())
    assert scene2.scene_id != scene.scene_id
    assert scene2.asset_id != scene.asset_id


def test_reconstruction_import_confined(tmp_path, tour_root):
    output_dir = tmp_path / "recon"
    subject_dir = output_dir / "oksubj"
    subject_dir.mkdir(parents=True)
    points, colors = make_points(count=300)
    write_ascii_ply(subject_dir / "pointcloud.ply", points, colors)
    repository = TourRepository(tour_root)
    scene = import_from_reconstruction(repository, "oksubj", output_dir, "重建ok", make_config())
    assert scene.source_kind == "real"
    # traversal is rejected
    with pytest.raises(InvalidDataError):
        import_from_reconstruction(repository, "../evil", output_dir, None, make_config())
    # unknown subject
    with pytest.raises(NotFoundError):
        import_from_reconstruction(repository, "nope", output_dir, None, make_config())


def test_sanitize_name():
    assert sanitize_name("a/b\\c 中") == "a_b_c_中"
    assert sanitize_name("///") == "subject"


def test_seed_demo_full_loop(tmp_path, tour_root):
    repository = TourRepository(tour_root)
    result = seed_demo(repository, make_config())
    scene = repository.get_scene(result["scene_id"])
    assert scene.source_kind == "synthetic"
    assert len(scene.entities) == 3
    assert len(scene.views) == 4
    storyboard = repository.get_storyboard(result["storyboard_id"])
    assert len(storyboard.shots) == 3
    assert sum(s.duration_seconds for s in storyboard.shots) == 30
    from src.tour.validation import validate_storyboard

    validation = validate_storyboard(scene, storyboard, duration_strict=False)
    assert validation.status == "valid", validation.errors
    # second seed into the same root must refuse to overwrite
    from src.tour.repository import ConflictError

    with pytest.raises(ConflictError):
        seed_demo(repository, make_config())
