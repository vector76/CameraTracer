#%%
import traceback
import threading
import queue
import cv2
from .cvtrace import PathManager, GcodeWriter

class TracerWorker(threading.Thread):
    def __init__(self, parent):
        # Daemon but we'll still try to shut down nicely
        threading.Thread.__init__(self, daemon=True)
        self.inbox = queue.Queue()
        self.parent = parent
        self.xyz = None
        
        
    def run(self):
        while True:
            message = self.inbox.get()  # wait until message is available
            cmd = message[0]
            self.log_info(f"Got command {cmd}")
            
            if cmd == "start_test":
                self.test_routine_1()
                self.log_info("Worker done with test_routine_1")
            elif cmd == "start_test2":
                self.test_routine_2()
                self.log_info("Worker done with test_routine_2")
            elif cmd == "start_test3":
                self.test_routine_3()
                self.log_info("Worker done with test_routine_3")
            elif cmd == "cancel":
                self.log_info("Got cancel request while not running, ignoring")
            elif cmd == "continue":
                self.log_info("Got continue request while not running, ignoring")
            elif cmd == "position":
                self.log_info("Got position request while not running, ignoring")
            elif cmd == "shutdown":
                self.log_info("Got shutdown request while not running")
                break
            else:
                self.log_info(f"Worker got unrecognized command: {cmd}")
        self.log_info("Done running")


    def delay_command(self):
        delay_ms = round(self.parent._settings.getFloat(["scan_delay"]) * 1000)
        return f"G4 P{delay_ms}"
    

    def wait(self, delay):
        self.commands("G4 P1")
        self.commands("M117 TRACER")
        while True:
            try:
                message = self.inbox.get(timeout=delay)
            except queue.Empty:
                message = ("timeout",)  # timed out waiting for message
                
            cmd = message[0]
            if cmd == "cancel":
                self.log_info("Worker received cancel request")
                raise Exception("cancel")
            elif cmd == "timeout":
                self.log_info("Worker encountered timeout waiting")
                raise Exception("timeout")
            elif cmd == "shutdown":
                self.log_info("Worker received shutdown request")
                self.post(("shutdown",))
                raise Exception("shutdown")
            elif cmd == "continue":
                self.log_info("Worker continuing")
                return
            elif cmd == "position":
                if self.xyz is None:
                    self.xyz = (message[1], message[2], message[3])
                else:
                    self.log_info("Worker got unexpected xyz, ignoring")
                # repeat loop, process more messages
            else:
                self.log_info(f"Worker got unexpected command {cmd}, ignoring")
                # repeat loop, process more messages


    def test_routine_1(self):
        try:
            self.xyz = None
            self.commands("M114")
            self.commands("G90")
            self.wait(5)
            if self.xyz is None:
                self.log_info("Worker did not get xyz, aborting")
                return
            xyz = self.xyz  # keep local copy
            self.commands(f"G1 X{xyz[0]+10} F800")
            self.wait(5)
            self.commands(f"G1 Y{xyz[1]+10} F800")
            self.wait(5)
            self.commands(f"G1 X{xyz[0]} F800")
            self.wait(5)
            self.commands(f"G1 Y{xyz[1]} F800")
            self.wait(5)
        except Exception as e:
            self.log_info(f"Worker got exception {e}, aborting")
    
    
    def test_routine_2(self):
        try:
            data_folder = self.parent.get_plugin_data_folder()
            self.xyz = None
            self.commands("M114")
            self.commands("G90")
            self.wait(5)
            if self.xyz is None:
                self.log_info("Worker did not get xyz, aborting")
                return
            xyz = self.xyz  # keep local copy
            pm = PathManager()
            img1 = self.parent.getImage()
            cv2.imwrite(f"{data_folder}/img1.png", img1)
            self.commands(f"G1 X{xyz[0]+10} F800")
            self.commands(self.delay_command())
            self.wait(5)
            img2 = self.parent.getImage()
            cv2.imwrite(f"{data_folder}/img2.png", img2)
            self.commands(f"G1 X{xyz[0]} Y{xyz[1]+10} F800")
            self.commands(self.delay_command())
            self.wait(5)
            img3 = self.parent.getImage()
            cv2.imwrite(f"{data_folder}/img3.png", img3)
            pm.addCapture(img1, xyz[0], xyz[1])
            pm.addCapture(img2, xyz[0]+10, xyz[1])
            st, nextx, nexty = pm.addCapture(img3, xyz[0], xyz[1]+10)
            self.commands(f"G1 X{nextx:.2f} Y{nexty:.2f} F800")
            self.commands(self.delay_command())
            self.wait(5)
            img4 = self.parent.getImage()
            cv2.imwrite(f"{data_folder}/img4.png", img4)
            self.log_info("done")
        except Exception as e:
            self.log_info(f"Worker got exception {e}, aborting")
    
    
    def test_routine_3(self):
        try:
            data_folder = self.parent.get_plugin_data_folder()
            self.xyz = None
            self.commands("M114")
            self.commands("G90")
            self.wait(5)
            if self.xyz is None:
                self.log_info("Worker did not get xyz, aborting")
                return
            xyz = self.xyz  # keep local copy
            pm = PathManager()
            img = self.parent.getImage()
            imgcount = 1
            cv2.imwrite(f"{data_folder}/img{imgcount}.png", img)
            st, nextx, nexty = pm.addCapture(img, xyz[0], xyz[1])
            while st == "more":
                if imgcount > 50:
                    self.log_info("exceeded capture limit, aborting")
                    return
                self.log_info(f"Capture more, next at {nextx} {nexty}")
                self.commands(f"G1 X{nextx:.2f} Y{nexty:.2f} F800")
                self.commands(self.delay_command())
                self.wait(5)
                img = self.parent.getImage()
                imgcount = imgcount + 1
                cv2.imwrite(f"{data_folder}/img{imgcount}.png", img)
                st, nextx, nexty = pm.addCapture(img, nextx, nexty)
            self.log_info("finalizing")
            
            # contour settings
            cset = dict(
                tool_offs_x = self.parent._settings.getFloat(["tool_offs_x"]),
                tool_offs_y = self.parent._settings.getFloat(["tool_offs_y"]),
                tool_diam = self.parent._settings.getFloat(["tool_diam"]),
                cut_offset = self.parent._settings.getFloat(["cut_offset"]),
                cut_hole = self.parent._settings.get_boolean(["cut_hole"]),
                cut_climb = self.parent._settings.get_boolean(["cut_climb"]),
            )

            # for get_contours, settings must have tool_offs_x, tool_offs_y, 
            # tool_diam, cut_offset, cut_hole (bool), cut_climb (bool)

            contours_xy = pm.get_contours(cset, outfolder=data_folder)
            self.log_info("completed contour extraction")
            
            # g-code generation settings
            gset = dict(
                cut_depth = self.parent._settings.getFloat(["cut_depth"]),
                cut_feedrate = self.parent._settings.getFloat(["cut_feedrate"]),
            )
            gw = GcodeWriter(contours_xy[0], gset)
            self.parent.lfs.add_file('generated.gcode', gw)
            self.log_info("done")
        except Exception:
            self.log_info(f"Worker got exception {traceback.format_exc()}, aborting")
    
    
    def commands(self, gcode):
        self.log_info(f"sending gcode to parent: {gcode}")
        self.parent._printer.commands(gcode)
    
    
    def log_info(self, string):
        self.parent._logger.info(string)
    
    
    def post(self, message):
        # someone else wants to post to my inbox
        self.inbox.put(message)