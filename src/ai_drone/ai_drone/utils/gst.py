from urllib.parse import urlparse

def _auth_props(rtsp: str) -> str:
    u = urlparse(rtsp)
    if u.username or u.password:
        user = (u.username or "").replace('"', '\\"')
        pw   = (u.password or "").replace('"', '\\"')
        return f'user-id="{user}" user-pw="{pw}" '
    return ""

def gst_pipelines(rtsp: str, use_hw: bool, width: int, height: int, latency: int):
    """
    Return a list of candidate GStreamer pipelines to try in order:
      1) TCP, HW decode (Jetson)
      2) TCP, SW decode
      3) UDP, HW decode
      4) UDP, SW decode
    All end in appsink drop=1 sync=false for low latency.
    """
    scale = "" if width <= 0 or height <= 0 else f" ! videoscale ! video/x-raw,width={width},height={height}"
    auth = _auth_props(rtsp)
    lat = max(50, int(latency))  # avoid 0

    hwdec = "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx"
    swdec = "avdec_h264"
    tail  = f" ! videoconvert ! video/x-raw,format=BGR{scale} ! appsink drop=1 max-buffers=1 sync=false"

    pipes = []
    # TCP
    pipes.append(
        f'rtspsrc location="{rtsp}" {auth} protocols=tcp latency={lat} drop-on-latency=true ! '
        f'rtph264depay ! h264parse ! ' + (hwdec if use_hw else swdec) + tail
    )
    pipes.append(
        f'rtspsrc location="{rtsp}" {auth} protocols=tcp latency={lat} drop-on-latency=true ! '
        f'rtph264depay ! h264parse ! ' + (swdec if use_hw else swdec) + tail
    )
    # UDP
    pipes.append(
        f'rtspsrc location="{rtsp}" {auth} protocols=udp latency={lat} drop-on-latency=true ! '
        f'rtph264depay ! h264parse config-interval=1 ! ' + (hwdec if use_hw else swdec) + tail
    )
    pipes.append(
        f'rtspsrc location="{rtsp}" {auth} protocols=udp latency={lat} drop-on-latency=true ! '
        f'rtph264depay ! h264parse config-interval=1 ! ' + (swdec) + tail
    )
    return pipes

