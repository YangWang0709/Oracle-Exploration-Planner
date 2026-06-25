from __future__ import annotations

from oracle_explorer.ros2.topics import multisensor_topic_config


def test_ros2_topic_config_generation() -> None:
    config = multisensor_topic_config(enable_rgb=True, enable_depth=True, enable_depth_pointcloud=True, enable_lidar=True, enable_scan=True)

    topics = config["topics_published"]
    assert "/camera/rgb/image_raw" in topics
    assert "/camera/depth/points" in topics
    assert "/lidar/points" in topics
    assert "/scan" in topics
    assert config["message_types"]["/scan"] == "sensor_msgs/msg/LaserScan"
