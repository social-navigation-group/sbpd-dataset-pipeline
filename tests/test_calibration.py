"""Calibration parsers, using excerpts of the files in sampled_data/extrinsics_intrinsics/."""

import numpy as np
import pytest

from acme_dataset.calibration import (calibration_dir, calibration_files, read_camera_info_txt,
                                      read_extrinsics_txt, read_static_transforms_yaml,
                                      read_tf_static_txt, transform_matrix)

CAMERA_INFO = """frame_id: zed_left_camera_optical_frame
size: 640 x 360
distortion_model: rational_polynomial
D (distortion): [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
K (intrinsic 3x3):
      [260.862061, 0.000000, 318.898834]
      [0.000000, 260.862061, 174.953232]
      [0.000000, 0.000000, 1.000000]
R (rectification 3x3):
      [1.000000, 0.000000, 0.000000]
      [0.000000, 1.000000, 0.000000]
      [0.000000, 0.000000, 1.000000]
P (projection 3x4):
      [260.862061, 0.000000, 318.898834, 0.000000]
      [0.000000, 260.862061, 174.953232, 0.000000]
      [0.000000, 0.000000, 1.000000, 0.000000]
"""

TF_STATIC = """  • local/base_link -> local/camera_link 
    Translation: [0.166, -0.003, 0.452] 
    Rotation (Quat): [0.000, 0.000, 0.000, 1.000] 
  • camera_depth_frame -> camera_depth_optical_frame 
    Translation: [0.000, -0.000, -0.000] 
    Rotation (Quat): [-0.500, 0.500, -0.500, 0.500] 
"""

STATIC_YAML = """# Robot static transforms
static_transforms:
  - parent: base_link
    child: rslidar
    translation: [0.14543, 0.0, 0.13312]
    rotation: [0.0, 0.0, 0.0, 1.0]
"""


@pytest.fixture
def calib_root(tmp_path):
    base = tmp_path / "sampled_data" / "extrinsics_intrinsics"
    (base / "cmu").mkdir(parents=True)
    (base / "cmu" / "_zed_zed_node_left_camera_info.txt").write_text(CAMERA_INFO)
    (base / "cmu" / "_tf_static.txt").write_text(TF_STATIC)
    traj = base / "nus" / "body_to_camera_optical_frame" / "t_0"
    traj.mkdir(parents=True)
    (base / "nus" / "static_transforms.yaml").write_text(STATIC_YAML)
    (traj / "extrinsics.txt").write_text("\n".join(str(v) for v in np.eye(4).reshape(-1)))
    (base / "nus" / "intrinsics.txt").write_text("321.29,0.0,321.98,0.0,320.53,175.45,0.0,0.0,1.0")
    return tmp_path


def test_files(calib_root):
    assert calibration_dir(calib_root, "go2nus").name == "nus"  # alias
    assert len(calibration_files(calib_root, "cmu")) == 2
    assert calibration_files(calib_root, "go2nus", "t_0")[-1].name == "extrinsics.txt"
    assert calibration_files(calib_root, "ubonn") == []


def test_parsers(calib_root):
    base = calib_root / "sampled_data" / "extrinsics_intrinsics"
    info = read_camera_info_txt(base / "cmu" / "_zed_zed_node_left_camera_info.txt")
    assert (info["width"], info["height"]) == (640, 360)
    assert info["K"][0, 2] == pytest.approx(318.898834) and info["D"][0] == 0.1
    assert read_camera_info_txt(base / "nus" / "intrinsics.txt")["K"][1, 1] == pytest.approx(320.53)
    tf = read_tf_static_txt(base / "cmu" / "_tf_static.txt")
    assert tf[["parent", "child"]].values.tolist() == [["local/base_link", "local/camera_link"],
                                                       ["camera_depth_frame", "camera_depth_optical_frame"]]
    np.testing.assert_allclose(transform_matrix(tf.iloc[0])[:3, 3], [0.166, -0.003, 0.452])
    optical = transform_matrix(tf.iloc[1])[:3, :3]
    np.testing.assert_allclose(optical @ [0, 0, 1], [1, 0, 0], atol=1e-9)  # optical z = body x
    assert read_static_transforms_yaml(base / "nus" / "static_transforms.yaml")["child"].tolist() == ["rslidar"]
    np.testing.assert_allclose(read_extrinsics_txt(base / "nus" / "body_to_camera_optical_frame" / "t_0" /
                                                   "extrinsics.txt"), np.eye(4))
