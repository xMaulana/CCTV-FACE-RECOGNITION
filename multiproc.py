import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="video_codec;h264_cuvid"

import cv2
from tools.camera import RTSPStream, RTSP_Server, compress_frame
import numpy as np
import multiprocessing as mp
from models.mtcnn import initialize_mtcnn
from tools.tools import sleep_ms, normalize_point, align_face, img2tensor, load_facebank, crop_face, get_datetime, str_to_datetime
import datetime as dt
from tools.database import Database
from models.arcface import initialize_arcface, BACKBONE
import keyboard
import traceback
import copy
import torch
import yaml

from tools.detect_face import detect_face, batching_detect_face
from tools.config import REFERENCE_POINT

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
YOLO_MARGIN = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"

def camera_worker(cctv_data, id_process, frame_queue:mp.Queue, proc_queue:dict, stop_event):
    cap = RTSPStream(cctv_data[1][0], dev=True).start()

    rtsp_server = None
    if cctv_data[1][2]:
        assert CONFIG["rtsp_server_host"] is not None
        rtsp_server = RTSP_Server(CONFIG["rtsp_server_host"],id_process).start()

    allow_put = True
    temp_coor = []

    while True:
        # key = cv2.waitKey(1) & 0xFF

        frame = cap.read()

        if frame is None:
            frame = np.zeros((720,1080,3))

        temp_frame = copy.deepcopy(frame)

        if allow_put:
            frame = compress_frame(frame, 80)
            frame_queue.put((id_process, frame, cctv_data[0]))
            allow_put = False
        else:
            if not proc_queue[id_process].empty():
                res = proc_queue[id_process].get()
                temp_coor = res
                allow_put = True
        
        for i in temp_coor:
            x1, y1, x2, y2 = i["bbox"]
            labl = f"{i['class']}:{i['distance']:.2f}"

            for j in i["landmarks"]:
                cv2.circle(temp_frame, j, 1, (0,255,0), -1)
                
            cv2.rectangle(temp_frame, (x1,y1), (x2, y2), (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(temp_frame, labl, (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2, cv2.LINE_AA)


        # cv2.imshow(str(id_process), frame)
        if rtsp_server is not None:
            rtsp_server.write_frame(temp_frame)

        sleep_ms(20)

        # if key == ord("q"):
        #     break
        
        if stop_event.is_set():
            break
    if rtsp_server is not None:
        rtsp_server.stop()
    cap.stop()

def detect_recog(frame_queue:mp.Queue, proc_queue:dict, database_queue:mp.Queue, check_dict:dict, stop_event):
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    if stream is not None:
        torch.cuda.set_stream(stream)
    
    mtcnn = initialize_mtcnn(device, selection_method="probability", keep_all=True)
    arcface = initialize_arcface(device=device, bbone=BACKBONE.IRESNET100)
    facebank = load_facebank(CONFIG["facebank_output"], device)
    
    while True:
        try:
            if stop_event.is_set():
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                    sleep_ms(1)
                
                for i in proc_queue.values():
                    while not i.empty():
                        i.get_nowait()
                        sleep_ms(1)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

            if not frame_queue.empty():
                proc_id, frame, cctv_id = frame_queue.get()
                frame = np.frombuffer(frame, np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                result = batching_detect_face(frame, mtcnn)
                new_result = []

                for x in result:
                    val_det = x
                    val_det["distance"] = 9.99
                    val_det["class"] = "UNKNOWN"
                    tmpres = crop_face(frame, val_det["bbox"])
                    normalized_point = normalize_point(val_det["landmarks"], val_det["bbox"])
                    aligned_face = align_face(tmpres, normalized_point, REFERENCE_POINT)
                    tensor_face = img2tensor(aligned_face,device=device)
                    with torch.no_grad():
                        embed = arcface(tensor_face).to(device, non_blocking=True)
                        dtect = torch.nn.functional.pairwise_distance(embed, facebank["features"])
                        val_det["distance"] = dtect.min().detach().cpu().numpy()
                        if val_det["distance"] < 0.8:
                            val_det["class"] = facebank["classes"][dtect.argmin().detach().cpu().numpy()]

                            worker_id = val_det["class"].split("_")[0]
                            worker_name = val_det["class"].split("_")[-1]
                            changed = False
                            if not worker_id in check_dict.keys():
                                check_dict[worker_id] = {
                                    "name": worker_name,
                                    "location": cctv_id,
                                    "timestamp": get_datetime()
                                }
                                changed = True
                            elif (check_dict[worker_id]["location"] != cctv_id) or (dt.datetime.timestamp(get_datetime(string=False)) - dt.datetime.timestamp(str_to_datetime(check_dict[worker_id]["timestamp"])) > 3600):
                                check_dict[worker_id] = {
                                    "name": worker_name,
                                    "location": cctv_id,
                                    "timestamp": get_datetime()
                                }
                                changed = True

                            if changed:
                                id_cctvn = cctv_id.rsplit("_")
                                database_queue.put((worker_id, worker_name, id_cctvn[1],cctv_id, id_cctvn[2], id_cctvn[0], check_dict[worker_id]["timestamp"]))
                    if CONFIG["sources"][cctv_id][2]:    
                        new_result.append(val_det)
                    sleep_ms(1)
                        
                proc_queue[proc_id].put(new_result)
            else:
                sleep_ms(10)
        except Exception as e:
            print(traceback.format_exc())

def database_worker(sql_config:dict, data:mp.Queue, stop_signal):
    database = Database(sql_config)

    while True:
        if stop_signal.is_set():
            break

        if not data.empty():
            to_push = data.get()
            database.insert(to_push)
        
        sleep_ms(10)

if __name__ == "__main__":
    print(f"USING {device.upper()}")
    mp.set_start_method("spawn", force=True)
    sources = list(CONFIG["sources"].items())
    to_allow_dict = mp.Manager().dict()

    database_queue = mp.Queue()
    frame_queue = mp.Queue()
    proc_queue = {x:mp.Queue() for x in range(len(sources))}
    stop_ev = mp.Event()
    
    db_worker = mp.Process(target=database_worker, args=(CONFIG["sql_config"], database_queue, stop_ev))
    db_worker.start()

    detector_workers = []
    for i in range(CONFIG["num_detectors"]):
        dr_proc = mp.Process(target=detect_recog, args=(frame_queue, proc_queue, database_queue, to_allow_dict, stop_ev))
        dr_proc.start()
        detector_workers.append(dr_proc)

    camera_workers = []
    for idx,val in enumerate(sources):
        proc = mp.Process(target=camera_worker, args=(val,idx, frame_queue, proc_queue, stop_ev))
        proc.start()
        camera_workers.append(proc)

    keyboard.wait("esc")
    stop_ev.set()

    if True:
        ####Force
        print("Done")
        os.system("taskkill /f /im python.exe")

    for i,p in enumerate(camera_workers):
        p.join()
        print(f"{i+1} of {len(camera_workers)} camera workers stopped")

    for i,p in enumerate(detector_workers):
        p.join()
        print(f"{i+1} of {len(detector_workers)} detection workers stopped")

    db_worker.join()
    print("Database worker stopped")

    cv2.destroyAllWindows()
    print("Done")