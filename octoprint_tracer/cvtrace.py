# -*- coding: utf-8 -*-

#%% Module for doing curve tracing with OpenCV
import numpy as np
import cv2
import math

#%% Arrowboard copied from test6
class ArrowBoard:
    def __init__(self):
        self.board_rows = 7
        self.board_cols = 7
        # triplets of id, row, col (row and col are zero-based)
        self.pattern = [(1, 0, 0), 
                        (2, 0, 1),
                        (3, 0, 2),
                        (4, 0, 3),
                        (5, 1, 0),
                        (6, 1, 1),
                        (7, 2, 0),
                        (8, 2, 2),
                        (9, 3, 0),
                        (10, 3, 3),
                        (11, 4, 4),
                        (12, 5, 5),
                        (13, 6, 6)]
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.code_size = 6  # includes only the interior of the black square
        self.mark_size = self.code_size+2  # includes the black square
        self.mark_pitch = self.mark_size+2  # includes white outer perimeter
    
    def insertMark(self, board, mark_id, r, c):
        # start and end indicate black portions, not included in pitch
        rstart = r + 1
        rend = r + self.mark_pitch - 1
        cstart = c + 1
        cend = c + self.mark_pitch - 1
        mark = cv2.aruco.drawMarker(self.dict, mark_id, self.mark_size)
        board[rstart:rend, cstart:cend] = mark
        return board
    
    # generate small board with one pixel per module
    def generate(self):
        row_modules = self.board_rows * self.mark_pitch
        col_modules = self.board_cols * self.mark_pitch
        
        # start with white background
        board = np.ones((row_modules, col_modules), dtype=np.uint8)*255
        for tag in self.pattern:
            board = self.insertMark(board, tag[0], tag[1]*self.mark_pitch, tag[2]*self.mark_pitch)
        
        return board
    
    def generate_96dpi(self):
        smallboard = self.generate()  # generated at one pixel per module
        # each tag at 1/3 inch means at 96 dpi, each tag is 32 pixels
        board96 = cv2.resize(smallboard, (32*self.board_cols, 32*self.board_rows), interpolation=cv2.INTER_NEAREST)
        return board96

    def findPoint(self, img):
        para = cv2.aruco.DetectorParameters_create()
        para.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        found = cv2.aruco.detectMarkers(img, self.dict, parameters=para)
        return self.extrapolatePoint(found)

    def extrapolatePoint(self, found):
        # found is presumed to be the result of detectMarkers, in which case
        # found[0] is rectangles of tags
        # found[1] is identities of tags
        nearest = None
        origin = None
        upleft = None
        for rect, ident in zip(found[0], found[1]):
            t_origin, t_dist, t_ul = self.extrapolateSingle(rect, ident)
            if t_dist is not None:
                if nearest is None or t_dist < nearest:
                    origin = t_origin
                    nearest = t_dist
                    upleft = t_ul
        return origin, upleft

    def extrapolateSingle(self, rect, ident):
        r = rect[0]
        mid = np.average(r, 0)
        offs = r - mid
        offs_full = offs*self.mark_pitch/self.mark_size
        # to take one step up or left relative to the tag
        up = 0.5*(offs_full[0] - offs_full[3] + offs_full[1] - offs_full[2])
        left = 0.5*(offs_full[0] - offs_full[1] + offs_full[3] - offs_full[2])
        # locate the triplet within self.pattern that matches the identity
        tagpos = next((x for x in self.pattern if x[0] == ident), None)
        if tagpos is None:
            return None, None
        
        upleft = up + left
        upsteps = tagpos[1] + 0.5
        leftsteps = tagpos[2] + 0.5
        origin = mid + upsteps*up + leftsteps*left
        distance = np.linalg.norm(upsteps*up + leftsteps*left)
        # print(f"extrapolated origin {origin} at distance {distance} from tag {ident}")
        return origin, distance, upleft


#%% Given image, and point (column, row), find nearest nonzero pixel (column, row)
def nearestNonzero(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index][0]

#%% Line filter, given grayscale image and approximate line width in pixels, filter line
def lineFilter(grayimg, linew_pixels):
    ksize = round(linew_pixels*1.5)*2+1  # about 3 times line width
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    back = cv2.morphologyEx(grayimg, cv2.MORPH_CLOSE, k2)  # background color
    imn = (grayimg / back * 200).astype("uint8")  # normalized to background = 200
    _, line = cv2.threshold(imn, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
    return line

#%%
class PathCalculator:
    # requires three images to initialize
    def __init__(self, base, plusx, plusy, stepsize_mm, linew_mm = 1.0):
        self.linew_mm = linew_mm
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        para = cv2.aruco.DetectorParameters_create()
        para.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        found1 = cv2.aruco.detectMarkers(base, aruco_dict, parameters=para)
        found2 = cv2.aruco.detectMarkers(plusx, aruco_dict, parameters=para)
        found3 = cv2.aruco.detectMarkers(plusy, aruco_dict, parameters=para)
        
        img_center = (base.shape[1]/2, base.shape[0]/2)    
        dxvec = getDisplacement(found2, found1, img_center)/stepsize_mm
        dyvec = getDisplacement(found3, found1, img_center)/stepsize_mm
        
        ppmmx = np.linalg.norm(dxvec)
        ppmmy = np.linalg.norm(dyvec)
        if ppmmx/ppmmy > 1.1 or ppmmy/ppmmx > 1.1:
            raise Exception("Inconsistent pixels per mm in x and y")
            
        self.ppmm = math.sqrt(ppmmx*ppmmy)
        
        # angle of x direction, counter clockwise from horizontal/right
        xang = math.atan2(-dxvec[1], dxvec[0])*180/math.pi
        yang = math.atan2(-dyvec[0], -dyvec[1])*180/math.pi
        
        # don't freak out if xang=179.9 and yang = -179.9, they differ by 0.2
        delta_ang = (yang-xang+180)%360 - 180
        
        if abs(delta_ang) > 2:
            raise Exception("Inconsistent angle of x and y axes")
            
        self.ang = (xang + delta_ang/2 + 180)%360 - 180
        
        self.unified = None
        self.cornerpos_xy = None  # upper left (positive y, negative x) corner
        self.mask = None
        
    def addImage(self, image, centerpos_xy, dorotate=True):
        if dorotate:
            c, m = cropAndMask(rotateImage(image, -self.ang))
        else:
            c, m = cropAndMask(rotateImage(image, 0))

        xr_mm = c.shape[1]/2/self.ppmm  # image radius in x dimension
        yr_mm = c.shape[0]/2/self.ppmm  # image radius in y dimension
        # upper left corner position of the cropped masked fragment
        im_cpos_xy = centerpos_xy + np.array([-xr_mm, yr_mm])
        
        if self.unified is None:
            self.unified = c
            self.mask = m
            self.cornerpos_xy = im_cpos_xy
        else:
            # need to merge new image with existing
            iterations = 0
            while True:  # emulate do-while
                # location of fragment corner relative to existing image
                rel_im_cpos_xy = im_cpos_xy - self.cornerpos_xy
                # offset in colums, rows
                rel_im_cpos_cr = (round(rel_im_cpos_xy[0]*self.ppmm), round(-rel_im_cpos_xy[1]*self.ppmm))
                # rel_im_cpos_cr = np.round(rel_im_cpos_xy * self.ppmm)
                # first and last+1 rows and columns where new image would land
                #box = (int(rel_im_cpos_cr[1]), int(rel_im_cpos_cr[1]+c.shape[1]), int(rel_im_cpos_cr[0]), int(rel_im_cpos_cr[0]+c.shape[0]))
                box = (rel_im_cpos_cr[1], rel_im_cpos_cr[1]+c.shape[1], rel_im_cpos_cr[0], rel_im_cpos_cr[0]+c.shape[0])
                #print(f"iter {iterations+1}  box: {box},  unified shape: {self.unified.shape}")
                # calculate how far outside we would extend on all four sides
                etop = max(-box[0], 0)
                ebot = max(box[1]-self.unified.shape[0], 0)
                eleft = max(-box[2], 0)
                eright = max(box[3]-self.unified.shape[1], 0)
                if all([etop == 0, ebot == 0, eleft == 0, eright == 0]):
                    break  # no expansion necessary
                self.expandImages(etop, ebot, eleft, eright)
                iterations = iterations + 1
                if iterations > 10:
                    raise Exception("internal error, infinite loop in expanding")
                # after expanding, calculate again, roundoff could produce a tiny error and need expansion again
            
            uchunk = self.unified[box[0]:box[1], box[2]:box[3], :]
            mchunk = self.mask[box[0]:box[1], box[2]:box[3]]
            
            uchunk2, mchunk2 = weightCombine(c, m, uchunk, mchunk)
            self.unified[box[0]:box[1], box[2]:box[3], :] = uchunk2
            self.mask[box[0]:box[1], box[2]:box[3]] = mchunk2
            pass
        
        return c, m

    def expandImages(self, top, bottom, left, right):
        if any([top<0, bottom<0, left<0, right<0]):
            raise Exception("expansion amounts must be zero or positive")
        if not any([top>0, bottom>0, left>0, right>0]):
            #print(f"none worth expanding: {top}, {bottom}, {left}, {right}")
            return
        #print(f"expansions: {top}, {bottom}, {left}, {right}")
        #print(f"size before {self.unified.shape}")
        self.unified = cv2.copyMakeBorder(self.unified, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        #print(f"size after {self.unified.shape}")
        self.mask = cv2.copyMakeBorder(self.mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        dx = -left/self.ppmm
        dy = top/self.ppmm
        #print(f"dx, dy: ({dx}, {dy}) cornerpos before {self.cornerpos_xy}")
        self.cornerpos_xy = self.cornerpos_xy + np.array([dx, dy])
        #print(f"cornerpos after {self.cornerpos_xy}")
    
    def pixToXY(self, pixcol, pixrow):
        dxmm = pixcol/self.ppmm  # relative x position in mm from upper left
        dymm = -pixrow/self.ppmm # relative y posiiton in mm from upper left
        return self.cornerpos_xy + (dxmm, dymm)
    
    def findLine(self, img):
        img, mask = cropAndMask(rotateImage(img, -self.ang))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        linew_pixels = self.linew_mm * self.ppmm
        ksize = round(linew_pixels*1.5)*2+1  # about 3 times line width
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
        back = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k2)  # background color
        imn = (gray / (back+3) * 200).astype("uint8")  # normalized to background = 200
        _, line = cv2.threshold(imn, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
        line = line * (mask > 0)  # restrict to masked area
        # find pixel nearest center
        nnz_cr = nearestNonzero(line, (line.shape[1]/2, line.shape[0]/2))
        line2 = line.copy()
        cv2.floodFill(line2, None, (nnz_cr[0], nnz_cr[1]), 128)  # operates in place
        line = (line2 == 128).astype("uint8")*255
        return line, nnz_cr

    def findFirstPoint(self, samp, sampxy):
        line, nnz_cr = self.findLine(samp)
        #print(f"nnz_cr: {nnz_cr}")
        #print(f"shape/2: {line.shape[0]/2}")
        #print(f"sampxy: {sampxy}")
        #print(f"nnz adjusted from center: {(nnz_cr-line.shape[0]/2)}")
        first_xy = (nnz_cr-line.shape[0]/2) * (1/self.ppmm, -1/self.ppmm) + sampxy
        return first_xy

    def followLine(self, samp, sampxy, fwdxy, stopxy=None):
        line, nnz_cr = self.findLine(samp)
        contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = np.squeeze(contours[0])  # now dimensions are Nx2
        homedist = np.linalg.norm(cont-nnz_cr, None, 1)
        nearest_ix = np.argmin(homedist)
        pixsteps = round(self.linew_mm*self.ppmm+1)
        cplus = cont[(nearest_ix+pixsteps) % len(cont)]
        cminus = cont[(nearest_ix-pixsteps) % len(cont)]
        cdelta = cplus-cminus  # delta (columns, rows) in the forward (positive index) direction
        fwd_cr = (fwdxy[0], -fwdxy[1])
        csearch = np.concatenate((cont[nearest_ix+1:, :], cont[:nearest_ix, :]))
        if (fwd_cr * cdelta).sum() < 0:
            # negative increment on contour is forawrd direction on curve, flip it
            csearch = np.flipud(csearch)
        imgd = line.shape[0]
        cpos_cr = (imgd/2, imgd/2)  # columns, rows is reverse of shape
        cdist = np.linalg.norm(csearch-cpos_cr, None, 1)
        # find index where distance from center goes out of range
        out = next((i for i, x in enumerate(cdist) if x > imgd*0.4), None)
        stop = False
        if stopxy is not None:
            # find index where distance from stopxy is below 2*line width
            # convert stopxy into image-relative column and row
            stop_rel_xy = stopxy-sampxy # xy relative to center of image
            stopcr = stop_rel_xy * (self.ppmm, -self.ppmm) + cpos_cr
            sdist = np.linalg.norm(csearch-stopcr, None, 1)
            stopdist_pix = self.linew_mm*self.ppmm*2
            # find index where we are close to the stopping point
            stop_ix = next((i for i, x in enumerate(sdist) if x < stopdist_pix), None)
            if stop_ix is not None and stop_ix < out:
                # hit the stop point before next point on curve
                stop = True
        nextoffs_cr = csearch[out] - imgd/2
        cplus = csearch[(out+pixsteps) % len(csearch)]
        cminus = csearch[(out-pixsteps) % len(csearch)]
        cdelta = cplus-cminus  # delta (columns, rows) in the forward (positive index) direction
        next_xy = sampxy + nextoffs_cr * (1, -1)/self.ppmm
        nextfwd_xy = cdelta * (1, -1)
        return line, next_xy, nextfwd_xy, stop

    def findStartingPoint(self):
        ab = ArrowBoard()
        point, ori = ab.findPoint(self.unified)
        #print(f"vertex point: {point}  ori: {ori}")
        xypoint = self.pixToXY(point[0], point[1])
        xypoint2 = self.pixToXY(point[0]+ori[0], point[1]+ori[1])
        #print(f"xy coord (mm) of arrow point: {xypoint}  and ahead of point: {xypoint2}")
        #print(f"G1 X{xypoint2[0]:.2f} Y{xypoint2[1]:.2f} F300")
        fw_dir_xy = xypoint2 - xypoint
        right_dir_xy = (fw_dir_xy[1], -fw_dir_xy[0])
        return xypoint2, right_dir_xy
                
#%%
def weightCombine(im1, mask1, im2, mask2):
    i1f = im1.astype('float32')
    m1f = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB).astype('float32')/255
    i2f = im2.astype('float32')
    m2f = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB).astype('float32')/255
    wsum = i1f*m1f + i2f*m2f
    w = m1f+m2f
    wz = (w == 0).astype('float32')
    imc = wsum / (w + wz)
    msc = cv2.min(w[:, :, 1], 1)
    imc8 = imc.astype('uint8')
    msc8 = (msc*255).astype('uint8')
    return imc8, msc8
    

def cropAndMask(image):
    nr, nc = image.shape[:2]
    if nr < nc:
        rm = (nc-nr)/2
        if rm%1 != 0:
            raise Exception("rows and columns expected to be even")
        rm = round(rm)
        imgcrop = image[:, rm:-rm, :]
        n = nr
    elif nc < nr:
        rm = (nr-nc)/2
        if rm%1 != 0:
            raise Exception("rows and columns expected to be even")
        rm = round(rm)
        imgcrop = image[rm:-rm, :, :]
        n = nc
    else:
        imgcrop = image
        n = nr
    
    x = np.tile(np.linspace(-(n-1)/2, (n-1)/2, n), (n, 1))
    y = np.transpose(x)
    r = np.sqrt(x**2 + y**2)  # radius of each pixel
    w = 1-r/(n/2)
    mask = (cv2.min(cv2.max(w*4, 0), 1)*255).astype('uint8')
    nzmask = (mask > 0).astype('uint8')  # keep pixels with nonzero weight
    if image.ndim == 3:
        return imgcrop*cv2.cvtColor(nzmask, cv2.COLOR_GRAY2RGB), mask
    else:
        # assume two dimensions, not 1 or 4 or more
        return imgcrop*nzmask, mask


def rotateImage(image, angle):
    nr, nc = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((nc/2, nr/2), angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (nc,nr), flags=cv2.INTER_LINEAR)
    return new_image


def getDisplacement(tags1, tags2, center_col_row):
    best_mdist = None
    answer = None
    for id1, loc1 in zip(tags1[1], tags1[0]):
        for id2, loc2 in zip(tags2[1], tags2[0]):
            if id1 == id2:
                avgd = np.average(loc2-loc1, (0,1))
                midvec = np.average((loc1+loc2)/2-center_col_row, (0,1))
                mdist = np.linalg.norm(midvec)
                if answer is None or mdist < best_mdist:
                    best_mdist = mdist
                    answer = avgd
                break  # assume only one matching id2 for each id1
    return answer

#%%
class PathManager:
    def __init__(self):
        self.start_step = 10
        self.pc = None
        self.capturelist = []
        self.fwd_dir = None
        self.startstop_xy = None
    
    def addCapture(self, image, xpos, ypos):
        self.capturelist.append((image, xpos, ypos))
        lcl = len(self.capturelist)
        if lcl == 1:
            # got one, now capture second image at plus x
            status = 'more'
            next_x = xpos+self.start_step
            next_y = ypos
        elif lcl == 2:
            # got two, now capture third image at plus y
            orig_x = self.capturelist[0][1]
            orig_y = self.capturelist[0][2]
            status = 'more'
            next_x = orig_x
            next_y = orig_y+self.start_step
        elif lcl == 3:
            # got three, now initialize pc
            base = self.capturelist[0][0]
            plusx = self.capturelist[1][0]
            plusy = self.capturelist[2][0]
            self.pc = PathCalculator(base, plusx, plusy, self.start_step)
            # now that we have orientation and scale, insert the images to unify
            for i in range(3):
                self.pc.addImage(self.capturelist[i][0], (self.capturelist[i][1], self.capturelist[i][2]))
            nextpt, nextdir = self.pc.findStartingPoint()
            self.start_fwd_dir = nextdir  # save for later
            self.fwd_dir = nextdir

            self.pc.unified = None
            self.pc.cornerpos_xy = None
            self.pc.mask = None
            
            status = 'more'
            next_x = nextpt[0]
            next_y = nextpt[1]
        elif lcl == 4:
            # took this picture purely based on the arrow target, just find nearest point and go again
            self.startstop_xy = self.pc.findFirstPoint(image, (xpos, ypos))
            status = 'more'
            next_x = self.startstop_xy[0]
            next_y = self.startstop_xy[1]
        else:
            # fifth or later
            if lcl == 5:
                lineimg, newsamp, newfwd, stop = self.pc.followLine(image, (xpos, ypos), self.fwd_dir, None)  # no stop condition
            else:
                lineimg, newsamp, newfwd, stop = self.pc.followLine(image, (xpos, ypos), self.fwd_dir, self.startstop_xy)
                
            self.pc.addImage(cv2.cvtColor(lineimg, cv2.COLOR_GRAY2BGR), (xpos, ypos), dorotate=False)
            self.fwd_dir = newfwd
            if stop:
                status = 'stop'
            else:
                status = 'more'
            next_x = newsamp[0]
            next_y = newsamp[1]

        print(f"G1 X{next_x:.2f} Y{next_y:.2f}")
        print(f'pm.addCapture(cv2.imread("a{lcl+1}.jpg"), {next_x:.2f}, {next_y:.2f})')
        return status, next_x, next_y
    
    def get_contours(self, settings, outfolder=None):
        # settings must have tool_offs_x, tool_offs_y, tool_diam, cut_offset,
        # cut_hole (bool), cut_climb (bool)
        # cut_depth and cut_feedrate are not needed 

        t1 = self.pc.unified
        _, t2 = cv2.threshold(t1, 128, 255, cv2.THRESH_BINARY)  # threshold
        t3 = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY)
        t4 = cv2.ximgproc.thinning(t3)
        # cv2.imwrite("t86_thinned.png", t4)
        #startstop_rc = (self.startstop_xy - self.pc.cornerpos_xy) * (self.pc.ppmm, -self.pc.ppmm)
        t5 = t4.copy()
        cv2.floodFill(t5, None, (0,0), 255)
        # cv2.imwrite("t86_outside.png", t5)
        if settings['cut_hole']:
            inward = settings['tool_diam']/2 + settings['cut_offset']
            cw = not settings['cut_climb']
        else:
            inward = -settings['tool_diam']/2 - settings['cut_offset']
            cw = settings['cut_climb']
        # cw indicates if we cut clockwise around perimeter or not

        # print(f"inward: {inward}")
        if inward > 0:
            kern = diskStructuringElement(inward*self.pc.ppmm)
            # dilate to shrink inward
            adjusted = cv2.morphologyEx(t5, cv2.MORPH_DILATE, kern)
        elif inward < 0:
            kern = diskStructuringElement(-inward*self.pc.ppmm)
            # erode to enlarge outward
            adjusted = cv2.morphologyEx(t5, cv2.MORPH_ERODE, kern)
        else:
            adjusted = t5.copy()
        
        # cv2.imwrite("t86_adjusted.png", adjusted)
        
        if not cw:
            adjusted = 255-adjusted
            
        if outfolder is not None:
            cv2.imwrite(f"{outfolder}/finalized.png", adjusted)
        
        contours, _ = cv2.findContours(adjusted, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        contours_xy = []
        cp_xy = self.pc.cornerpos_xy  # shape (2,)
        to_xy = (settings["tool_offs_x"], settings["tool_offs_y"])
        for c in contours:
            # pixel coorindates to absolute x y mm coordinates
            contours_xy.append(np.squeeze(c) * (1/self.pc.ppmm, -1/self.pc.ppmm) + cp_xy + to_xy)
            
        cim = adjusted * 0
        for p in contours[0]:
            cim[p[0][1], p[0][0]] = 255  # row, column
        if outfolder is not None:
            cv2.imwrite(f"{outfolder}/contour_vertices.png", cim)

        return contours_xy
        
#%%
class GcodeWriter:
    def __init__(self, xy, settings):
        self.xy = xy
        self.cut_feedrate = settings["cut_feedrate"]
        self.cut_depth = settings["cut_depth"]
    
    def save(self, path):
        with open(path, 'w') as writer:
            fr = round(self.cut_feedrate)
            zfr = round(self.cut_feedrate/5)
            depth = -self.cut_depth
            pm1 = self.xy[-1,:]  # point "minus one"
            writer.write(f"G0 X{pm1[0]:.3f} Y{pm1[1]:.3f} F{fr}\n")
            writer.write(f"G1 Z{depth:.3f} F{zfr}\n")
            for p in self.xy:
                writer.write(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{fr}\n")
            writer.write(f"G0 Z0 F{zfr}\n")

#%%
def diskStructuringElement(radius):
    irad = math.floor(radius)
    y,x = np.ogrid[-irad:irad+1, -irad:irad+1]
    return (x**2 + y**2 <= radius**2).astype("uint8")

    
    