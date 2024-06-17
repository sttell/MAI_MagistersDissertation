import pyrealsense2 as rs


class Camera:
    """
    Class for RealSense camera implementation.
    """
    def __init__(self, settings):
        self.settings = settings
        self.ctx = rs.context()
        self.pipeline = rs.pipeline(self.ctx)
        self.config = rs.config()
        self.colorizer = rs.colorizer()

        self.config.enable_stream(
            rs.stream.color,
            self.settings["RES"]["W"],
            self.settings["RES"]["H"],
            rs.format.bgr8,
            self.settings["FPS"]
        )

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.holes_fill, 3)

    def start(self):
        """
        Streaming start.
        """
        self.pipeline.start(self.config)

    def wait_for_frames(self):
        """
        Wait for new frames from stream.
        :return: frames set
        """
        return self.pipeline.wait_for_frames()

    def stop(self):
        """
        Stop streaming.
        """
        self.pipeline.stop()