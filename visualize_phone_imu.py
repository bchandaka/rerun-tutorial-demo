import json
from flask import Flask, request, session
import cv2
import threading
import time 
import rerun as rr
import argparse
import numpy as np

CLOSE_THREAD = False

def log_translation(xyz:list, namespace:str):
    rr.log(namespace, rr.Transform3D(
        translation=rr.Vector3D(xyz=xyz),
    ))
def log_quaternion(q:list, namespace:str):
    rr.log(namespace, rr.Transform3D(
        rotation=rr.Quaternion(xyzw=q),
    ))
    rr.log(
        namespace,
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
def log_rgb(t:int, rgb:np.ndarray, camera_name:str):
    rr.set_time_nanos("record_time", t)
    rr.log(f'{camera_name}/rgb', rr.Image(rgb))
def log_phone_acc(t:int, x,y,z):
    rr.set_time_nanos("record_time", t)
    rr.log("iphone_acc/x", rr.TimeSeriesScalar(x))
    rr.log("iphone_acc/y", rr.TimeSeriesScalar(y))
    rr.log("iphone_acc/z", rr.TimeSeriesScalar(z))
def log_phone_orientation(t, qx,qy,qz,qw):
    rr.set_time_nanos("record_time", t)
    q = [qx, qy, qz, qw]
    log_quaternion(q, "ios/iphone/orientation")
def log_headphone(t:int, qx,qy,qz,qw, ax,ay,az):
    rr.set_time_nanos("record_time", t)
    q = [qx, qy, qz, qw]
    log_quaternion(q, "ios/headphone/orientation")
    rr.log("headphone_acc/x", rr.TimeSeriesScalar(ax))
    rr.log("headphone_acc/y", rr.TimeSeriesScalar(ay))
    rr.log("headphone_acc/z", rr.TimeSeriesScalar(az))
def log_watch(t:int, qx,qy,qz,qw, ax,ay,az):
    rr.set_time_nanos("record_time", t)
    q = [qx, qy, qz, qw]
    log_quaternion(q, "ios/watch/orientation")
    rr.log("watch_acc/x", rr.TimeSeriesScalar(ax))
    rr.log("watch_acc/y", rr.TimeSeriesScalar(ay))
    rr.log("watch_acc/z", rr.TimeSeriesScalar(az))

server = Flask(__name__)

def cleanup():
    global SAVE_ARG
    print("Cleaning up Server")

@server.route("/data", methods=["POST"])
def data():  # listens to the data streamed from the sensor logger
    global IS_RECORDING, dl, dv, VISUALIZE_ARG
    if str(request.method) == "POST":
        data = json.loads(request.data)
        for d in data['payload']:
            dataType = d.get("name", None)
            # print(f'dataType: {dataType}')
            if dataType == "accelerometer":
                values = d.get("values", None)
                if values is None:
                    continue
                values['time'] = d['time']
                log_phone_acc(values['time'], values['x'], values['y'], values['z'])
                IS_RECORDING = True
            elif dataType == "orientation":
                values = d.get("values", None)
                if values is None:
                    continue
                values['time'] = d['time']
                log_phone_orientation(values['time'], values['qx'], values['qy'], values['qz'], values['qw'])

            elif dataType == "headphone":
                values = d.get("values", None)
                if values is None:
                    continue
                values['time'] = d['time']
                log_headphone(values['time'], 
                    values['quaternionX'], values['quaternionY'], values['quaternionZ'], values['quaternionW'],
                    values['accelerationX'], values['accelerationY'], values['accelerationZ'])

            elif dataType == "wrist motion":
                values = d.get("values", None)
                if values is None:
                    continue
                values['time'] = d['time']
                log_watch(values['time'], 
                    values['quaternionX'], values['quaternionY'], values['quaternionZ'], values['quaternionW'],
                    values['accelerationX'], values['accelerationY'], values['accelerationZ'])
    return "success"

rgb_hz = 10
def vis_rgb():
    global CLOSE_THREAD
    vid = cv2.VideoCapture(0) 
    while not CLOSE_THREAD:
        color_t_ns = time.time_ns()
        ret, frame = vid.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        log_rgb(color_t_ns, frame, "webcam")
        time.sleep(1 / rgb_hz)
    vid.release()
    return 

def main() -> None:
    global CLOSE_THREAD
    parser = argparse.ArgumentParser(description="Streams frames from a connected realsense depth sensor.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    experiment_name = "rerun_iphone_imu"
    rr.script_setup(args, experiment_name)

    rr.init(experiment_name, spawn=True)
    rr.log("ios/watch", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log("ios/headphone", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log("ios/iphone", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)



    rgbd_thread = threading.Thread(target=vis_rgb)

    try:
        rgbd_thread.start()
        server.run(port=8000, host="0.0.0.0")
    finally:
        CLOSE_THREAD = True
        rgbd_thread.join()
        cleanup()
        rr.script_teardown(args)

if __name__ == "__main__":
    main()
