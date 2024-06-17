import subprocess as sp
import cv2

rtsp_url = 'rtsp://127.0.0.1:8554/video_stream'

video_path = 0


# We have to start the server up first, before the sending client (when using TCP). See: https://trac.ffmpeg.org/wiki/StreamingGuide#Pointtopointstreaming
ffplay_process = sp.Popen(['ffplay', '-rtsp_flags', 'listen', rtsp_url])  # Use FFplay sub-process for receiving the RTSP video.


cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frames width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frames height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video framerate

    
# FFmpeg command
command = ['ffmpeg',
           '-re',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-preset', 'ultrafast',
           '-f', 'rtsp',
           #'-flags', 'local_headers', 
           '-rtsp_transport', 'tcp',
           '-muxdelay', '0.1',
           '-bsf:v', 'dump_extra',
           rtsp_url]

p = sp.Popen(command, stdin=sp.PIPE)

while (cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    p.stdin.write(frame.tobytes())


p.stdin.close()  # Close stdin pipe
p.wait()  # Wait for FFmpeg sub-process to finish
ffplay_process.kill()  # Forcefully close FFplay sub-process