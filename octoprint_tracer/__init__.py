# coding=utf-8
from __future__ import absolute_import

import numpy as np
import cv2
import urllib
import flask
import octoprint.plugin
import octoprint.settings
import octoprint.filemanager.storage
import re
import io
import zipfile
import glob
import os
from .tracerworker import TracerWorker

__plugin_name__ = "Camera Tracer"
__plugin_pythoncompat__ = ">=3,<4"  # only python 3
__plugin_implementation__ = ""
__plugin_hooks__ = {}
__plugin_helpers__ = {}


# instantiate plugin object and register hook for gcode injection
def __plugin_load__():
    global __plugin_implementation__
    __plugin_implementation__ = TracerPlugin()

    global __plugin_hooks__
    __plugin_hooks__ = {
        "octoprint.comm.protocol.gcode.received": __plugin_implementation__.hook_gcode_received,
        "octoprint.comm.protocol.gcode.sending": __plugin_implementation__.hook_gcode_sending,
    }


class TracerPlugin(octoprint.plugin.StartupPlugin,
                   octoprint.plugin.ShutdownPlugin,
                   octoprint.plugin.TemplatePlugin,
                   octoprint.plugin.SettingsPlugin,
                   octoprint.plugin.AssetPlugin,
                   octoprint.plugin.BlueprintPlugin):
    
    def __init__(self):
        self.listening_for_pos = False
        self.t_start = None
        self.worker = TracerWorker(self)
        self.upload_folder = octoprint.settings.settings().getBaseFolder("uploads")
        self.lfs = octoprint.filemanager.storage.LocalFileStorage(self.upload_folder)

    def on_after_startup(self):
        self._logger.info(f"upload folder: {self.upload_folder}")
        self.worker.start()
        
    def on_shutdown(self):
        self.worker.post(("shutdown",))
        
    def get_settings_defaults(self):
        return dict(
            img_url="http://192.168.1.105/webcam/?action=snapshot",
            scan_delay = 0.5,
            tool_offs_x = 0,
            tool_offs_y = 0,
            tool_diam = 3.175,
            cut_offset = 0,
            cut_depth = 5,
            cut_feedrate = 300,
            cut_hole = True,
            cut_climb = True,
        )

    def get_template_configs(self):
        return [
            dict(type="settings", custom_bindings=False)
        ]

    def get_assets(self):
        return dict(
            js=["js/tracer.js"]
        )

    def getImage(self):
        resp = urllib.request.urlopen(self._settings.get(["img_url"]))
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    # let's see if I can figure this out
    @octoprint.plugin.BlueprintPlugin.route("/camera_image")
    def getCameraImage(self):
        image = self.getImage()
        # normalize and convert to monochrome
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(19,19))
        back = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k2)
        imn = (image / back * 200).astype(np.uint8)
        imnm = cv2.cvtColor(imn, cv2.COLOR_BGR2GRAY)
        is_success, pngbuffer = cv2.imencode('.png', imnm)
        return flask.Response(b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + pngbuffer.tobytes() + b'\r\n\r\n', 
                mimetype='multipart/x-mixed-replace; boundary=frame')

    @octoprint.plugin.BlueprintPlugin.route("/camera_skel")
    def getCameraSkel(self):
        image = self.getImage()
        # this structuring element must be larger than the lines we're looking for
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(19,19))
        # compute background by closing
        back = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k2)
        # compute 'normalized' image relative to background
        imn = (image / back * 200).astype(np.uint8)
        # normalized monochrome
        imnm = cv2.cvtColor(imn, cv2.COLOR_BGR2GRAY)
        # threshold and invert so detected line is white
        _, line = cv2.threshold(imnm, 150, 255, cv2.THRESH_BINARY_INV)
        # tiny closing to remove small defects
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        line2 = cv2.morphologyEx(line, cv2.MORPH_CLOSE, k3)
        # thinning
        thinned = cv2.ximgproc.thinning(line2)
        is_success, pngbuffer = cv2.imencode('.png', thinned)
        return flask.Response(b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + pngbuffer.tobytes() + b'\r\n\r\n', 
                mimetype='multipart/x-mixed-replace; boundary=frame')

    @octoprint.plugin.BlueprintPlugin.route("/camera_markers")
    def getCameraMarkers(self):
        image = self.getImage()
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        found = cv2.aruco.detectMarkers(image, aruco_dict)
        image_markers = cv2.aruco.drawDetectedMarkers(image, found[0], ids=found[1])
        is_success, pngbuffer = cv2.imencode('.png', image_markers)
        return flask.Response(b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + pngbuffer.tobytes() + b'\r\n\r\n', 
                mimetype='multipart/x-mixed-replace; boundary=frame')

    @octoprint.plugin.BlueprintPlugin.route("/camera_aim")
    def getCameraAim(self):
        image = self.getImage()
        nr = image.shape[0]
        nc = image.shape[1]
        image[round(nr/2),:,:] = 255
        image[:,round(nc/2),:] = 255
        is_success, pngbuffer = cv2.imencode('.png', image)
        return flask.Response(b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + pngbuffer.tobytes() + b'\r\n\r\n', 
                mimetype='multipart/x-mixed-replace; boundary=frame')

    @octoprint.plugin.BlueprintPlugin.route("/start_test")
    def startTest(self):
        self._logger.info("received /start_test request")
        if not self._printer.is_ready():
            return "not ready"
        self.worker.post(("start_test",))
        return "posted start_test"

    @octoprint.plugin.BlueprintPlugin.route("/start_test2")
    def startTest2(self):
        self._logger.info("received /start_test2 request")
        if not self._printer.is_ready():
            return "not ready"
        self.worker.post(("start_test2",))
        return "posted start_test2"

    @octoprint.plugin.BlueprintPlugin.route("/start_test3")
    def startTest3(self):
        self._logger.info("received /start_test3 request")
        if not self._printer.is_ready():
            return "not ready"
        self.worker.post(("start_test3",))
        return "posted start_test3"

    @octoprint.plugin.BlueprintPlugin.route("/cancel")
    def requestCancel(self):
        self.worker.post(("cancel",))
        return "cancel requested"

    @octoprint.plugin.BlueprintPlugin.route("/last_zip")
    def downloadZip(self):
        memory_file = io.BytesIO()
        data_folder = self.get_plugin_data_folder()
        pngfiles = glob.glob(f"{data_folder}/*.png")
        self._logger.info(f"got png files: {pngfiles}")
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for pfile in pngfiles:
                zf.write(pfile, arcname=os.path.basename(pfile))
        memory_file.seek(0)
        return flask.send_file(memory_file, attachment_filename='last.zip', as_attachment=True)

    def hook_gcode_sending(self, _comm, _phase, cmd, _cmd_type, gcode, *args, **kwargs):
        if cmd == "M117 TRACER":
            self.worker.post(("continue",))
            return (None,)  # suppress command
        if cmd == "M114":
            self.listening_for_pos = True
        return # return nothing, does nothing to alter or suppress command
        
    def hook_gcode_received(self, comm, line, *args, **kwargs):
        # Might look like this: "ok X:0.0 Y:0.0 Z:0.0 E:0.0 Count: A:0 B:0 C:0"
        if self.listening_for_pos:
            # only do this processing if we are expecting to catch the position, won't burden ordinary processing and reduced risk of false match
            matches = re.match("X:([0-9.-]+) Y:([0-9.-]+) Z:([0-9.-]+)", line)
            if matches:
                self.listening_for_pos = False
                self.worker.post(("position", float(matches.group(1)), float(matches.group(2)), float(matches.group(3))))
        
        # always return the response, regardless if we matched or not
        return line

