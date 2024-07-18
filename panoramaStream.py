#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  20 02:07:13 2019
python stream.py --device_id 0 --fps 30 --image_width 1920 --image_height 1080 --port 8554 --stream_uri /video_stream
@author: prabhakar
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
ffmpeg ffplay -i "rtsp://127.0.0.1:8554/video_stream"
"""
# import necessary argumnets 
import gi
import cv2
import argparse
import time
import sys
import os
import setting
staticBool = False

if staticBool:
    from run_static import init
else:
    from run_live import init


PORT = 8554
URL = '/video_stream'
WIDTH = setting.streamSize[0]
HEIGHT = setting.streamSize[1]
FPS = 20

now_dir = os.path.dirname(os.path.abspath(__file__))

# import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.names = ['front','back', 'left', 'right']
        names = ['front','back', 'left', 'right']
        paramsfile = [os.path.join("./my_yaml", name + ".yaml") for name in names]
        images = [os.path.join("./und_smimages", name + ".png") for name in names]
        weightsfile = "weights.png"
        maskfile = "masks.png"
        self.fisheyes,self.panorama,self.images = init(self.names,paramsfile,images,weightsfile,maskfile)
        self.number_frames = 0
        self.duration = 1 / FPS * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc bitrate=1200 speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(WIDTH, HEIGHT, FPS)
    # method to capture the video feed from the camera and push it to the
    # streaming buffer.
    def on_need_data(self, src, length):
        last = time.time()
        if staticBool:
            for f,name,img in zip(self.fisheyes,self.names,self.images):
                f.queue_in.put(img)
        frame =self.panorama.buffer.get()
        # It is better to change the resolution of the camera 
        # instead of changing the image shape as it affects the image quality.
        cv2.putText(frame, '%d' % (int(self.number_frames)), (637, 250),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_LINEAR)
       
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        print('pushed buffer, frame {}, size {}, runtime {}'.format(self.number_frames, len(data), time.time() - last))
        if retval != Gst.FlowReturn.OK:
            print(retval)
    # attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.set_service(str(PORT))
        self.get_mount_points().add_factory(URL, self.factory)
        self.attach(None)


# initializing the threads and running the stream on loop.
GObject.threads_init()
Gst.init(None)
server = GstServer()
loop = GObject.MainLoop()
loop.run()