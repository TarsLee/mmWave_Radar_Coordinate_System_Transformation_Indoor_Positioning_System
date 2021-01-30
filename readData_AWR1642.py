import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import cv2
import math
import time
tStart = time.time()

from scipy.optimize import curve_fit
import threading

# Change the configuration file name
configFileName = 'mmw_PC_14m_OfficeSpace.cfg'
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0;


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):

    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    #CLIport = serial.Serial('COM4', 115200)
    #Dataport = serial.Serial('COM3', 921600)
    print('check CLIportYYYYYYYYYYYYYYYYYYYY', CLIport.port)
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(splitWords[2])
            idleTime = int(splitWords[3])
            rampEndTime = int(float(splitWords[5]));
            freqSlopeConst = int(splitWords[8]);
            numAdcSamples = int(splitWords[10]);
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);


    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters

# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData16xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2**15;
    MMWDEMO_UART_MSG_POINT_CLOUD_2D = 6;
    MMWDEMO_UART_MSG_TARGET_LIST_2D = 7;
    MMWDEMO_UART_MSG_TARGET_INDEX_2D = 8;
    maxBufferSize = 2**15;
    tlvHeaderLengthInBytes = 8;
    pointLengthInBytes = 16;
    targetLengthInBytes = 68;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    targetObj = {}
    pointObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[20:20 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        timeStamp = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        chirpMargin = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        frameMargin = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        uartSentTime = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        trackProcessTime = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        word = [1, 2 ** 8]

        numTLVs = np.matmul(byteBuffer[idX:idX + 2], word)
        idX += 2
        checksum = np.matmul(byteBuffer[idX:idX + 2], word)
        idX += 2

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
        # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Initialize the tlv type
            tlv_type = 0

            try:
                # Check the header of the TLV message
                tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
            except:
                pass

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_POINT_CLOUD_2D:
                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                # Calculate the number of detected points
                numInputPoints = (tlv_length - tlvHeaderLengthInBytes) // pointLengthInBytes

                # Initialize the arrays
                rangeVal = np.zeros(numInputPoints, dtype=object)
                azimuth = np.zeros(numInputPoints, dtype=object)
                dopplerVal = np.zeros(numInputPoints, dtype=np.float32)
                snr = np.zeros(numInputPoints, dtype=np.float32)

                for objectNum in range(numInputPoints):
                    # Read the data for each object
                    rangeVal[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    azimuth[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    dopplerVal[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4
                    snr[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                    idX += 4

                    # Store the data in the detObj dictionary
                pointObj = {"numObj": numInputPoints, "range": rangeVal, "azimuth": azimuth,\
                            "doppler": dopplerVal, "snr": snr}

                #dataOK = 1

            elif tlv_type == MMWDEMO_UART_MSG_TARGET_LIST_2D:
                try:

                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

                    # Calculate the number of target points
                    numTargetPoints = (tlv_length - tlvHeaderLengthInBytes) // targetLengthInBytes

                    # Initialize the arrays
                    targetId = np.zeros(numTargetPoints, dtype=np.uint32)
                    posX = np.zeros(numTargetPoints, dtype=np.float32)
                    posY = np.zeros(numTargetPoints, dtype=np.float32)
                    velX = np.zeros(numTargetPoints, dtype=np.float32)
                    velY = np.zeros(numTargetPoints, dtype=np.float32)
                    accX = np.zeros(numTargetPoints, dtype=np.float32)
                    accY = np.zeros(numTargetPoints, dtype=np.float32)
                    EC = np.zeros((3, 3, numTargetPoints), dtype=np.float32)  # Error covariance matrix
                    G = np.zeros(numTargetPoints, dtype=np.float32)  # Gain

                    for objectNum in range(numTargetPoints):
                    # Read the data for each object
                        targetId[objectNum] = np.matmul(byteBuffer[idX:idX + 4], word)
                        idX += 4
                        posX[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        posY[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        velX[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        velY[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        accX[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        accY[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[0, 0, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[0, 1, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[0, 2, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[1, 0, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[1, 1, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[1, 2, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[2, 0, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[2, 1, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        EC[2, 2, objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4
                        G[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)
                        idX += 4

                    # Store the data in the detObj dictionary
                    targetObj = {"targetId": targetId, "posX": posX, "posY": posY, \
                                 "velX": velX, "velY": velY, "accX": accX, "accY": accY, \
                                 "EC": EC, "G": G}

                    dataOK = 1
                except:
                    pass

            elif tlv_type == MMWDEMO_UART_MSG_TARGET_INDEX_2D:
                # Calculate the length of the index message
                numIndices = tlv_length - tlvHeaderLengthInBytes
                indices = byteBuffer[idX:idX + numIndices]
                idX += numIndices


        # Remove already processed data
        if idX > 0:
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0


    return dataOK, frameNumber, targetObj, pointObj

# ------------------------------------------------------------------

#   ----- 雷達求對應b  --> 轉換成圖像的對應pixel  --> 旋轉角度得真正的pixel位置 -----   #
### 圖像的值    ###
fisheye_x_center = 1280
fisheye_y_center = 960

fisheye_x = [1280, 1293.527832, 1294.9313965, 1296.12854, 1292.380127, 1296.3099365]
fisheye_y = [960, 511.28717041, 383.63327026, 305.7442627, 266.3815918, 239.0788269]
#fisheye_y = [631.28717041, 488.63327026, 400.7442627, 346.3815918, 299.0788269]
#fisheye_y = [960, 651.28717041, 523.63327026, 435.7442627, 370.3815918, 323.0788269]
b_in_fisheye = []
for i in range(1, len(fisheye_x)):
    x_f = fisheye_x[i] - fisheye_x_center
    y_f = fisheye_y[i] - fisheye_y_center
    b_f = math.sqrt(x_f*x_f + y_f*y_f)
    b_in_fisheye.append(b_f)
for i in range(len(b_in_fisheye)):
    print(b_in_fisheye[i])

short_axis_in_fisheye = []
for i in range(1, len(fisheye_y)):
    short_axis_in_fisheye.append(int(fisheye_y_center-fisheye_y[i]))

long_axis_in_fisheye = []
for i in range(len(short_axis_in_fisheye)):
    long_axis_in_fisheye.append(int(1.2 *short_axis_in_fisheye[i]))

### 圖像的值    ###

### 雷達的值    ###
radar_x_center = -59.501004219055176
radar_y_center = 513.9549732208252

radar_x = [-115.00610113143921, -147.05049991607666, -182.19295740127563, -193.91080141067505, -219.43061351776123]
radar_y = [581.6828727722168, 613.6692047119141, 644.9289321899414, 678.2261371612549, 707.6416015625]
b_in_radar = []
for i in range(len(radar_x)):
    x_r = radar_x[i] - radar_x_center
    y_r = radar_y[i] - radar_y_center
    b_r = math.sqrt(x_r*x_r + y_r*y_r)
    b_in_radar.append(b_r)

### 雷達的值    ###

## curve fitting    ##
xdata = []
ydata = []
xdata.append(0)
ydata.append(960)
for i in range(len(b_in_radar)):
    xdata.append(b_in_radar[i])
for i in range(len(b_in_fisheye)):
    ydata.append(960 - b_in_fisheye[i])

def func(x, a, b, c, d):
    return a *pow(x,3) + b *pow(x,2) + c *x + d
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
## curve fitting    ##

def draw_coor(x, y):
    x_d = x - radar_x_center
    y_d = y - radar_y_center
    real_distace = math.sqrt((x_d*x_d)+(y_d*y_d))

    b_in_pixel = func(real_distace, popt[0], popt[1], popt[2], popt[3])
    b_in_pixel = 960 - b_in_pixel
    '''
    which_b_below = 0
    if(real_distace < b_in_radar[0]) :
        which_b_below = 0
    elif (real_distace >= b_in_radar[0] and real_distace < b_in_radar[1]) :
        which_b_below = 0
    elif (real_distace >= b_in_radar[1] and real_distace < b_in_radar[2]) :
        which_b_below = 1
    elif (real_distace >= b_in_radar[2] and real_distace < b_in_radar[3]) :
        which_b_below = 2
    elif (real_distace >= b_in_radar[3] and real_distace < b_in_radar[4]) :
        which_b_below = 3
    else:
        which_b_below = 3
    print(which_b_below)


    # b_in_radar, 0,1
    rate_in_radar = (real_distace-b_in_radar[which_b_below]) / (b_in_radar[which_b_below+1]-b_in_radar[which_b_below])
    b_in_pixel = rate_in_radar * (b_in_fisheye[which_b_below+1]-b_in_fisheye[which_b_below]) + b_in_fisheye[which_b_below]
    '''

    a = b_in_pixel
    radar_x = x - radar_x_center
    radar_y = y - radar_y_center
    pixel_x = (radar_x/real_distace) *a + 1280
    pixel_y = 960 - (radar_y/real_distace) *b_in_pixel

    return pixel_x, pixel_y

# ---------------    計算角度   ------------------------=

def angle(line_1, line_2):
    dx1 = line_1[2] - line_1[0]
    dy1 = line_1[3] - line_1[1]
    dx2 = line_2[2] - line_2[0]
    dy2 = line_2[3] - line_2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

# 角度(設為固定)=
AB_r = [-59.501004219055176, 513.9549732208252, -59.501004219055176, 600]
CD_r = [-59.501004219055176, 513.9549732208252, -115.00610113143921, 581.6828727722168]
ang1_r = angle(AB_r, CD_r)
# ---------------    計算選轉後的座標
# 計算sin, cos
sin_ang1 = math.sin(math.radians(ang1_r))
cos_ang1 = math.cos(math.radians(ang1_r))

# 計算出旋轉後座標
def coordinate_theta(x1, y1):
    x0 = 1280
    y0 = 960
    x_t = x1 - x0
    y_t = y1 - y0

    xp = x0 + (x_t * cos_ang1 - y_t * sin_ang1)
    yp = y0 + (x_t * sin_ang1 + y_t * cos_ang1)

    return xp, yp


# 最後, 雷達的x軸 和魚眼的x軸, 正負相反,需要調整
def x_pixel_contrast(x_pixel_org):
    org_delta = abs(x_pixel_org-fisheye_x_center)
    x_pixel_org_contrast = 0
    if (x_pixel_org-fisheye_x_center >= 0.0) :
        x_pixel_org_contrast = 1280 - org_delta
    else:
        x_pixel_org_contrast = 1280 + org_delta

    return x_pixel_org_contrast
# ---------------    計算角度   ------------------------
# ---------------    多執行緒, 用於接收影像   ------------------------
# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False

	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()

        self.capture.release()

URL = "rtsp://140.124.182.158/video2"

# 連接攝影機
ipcam = ipcamCapture(URL)

# 啟動子執行緒
ipcam.start()
# ---------------    多執行緒, 用於接收影像   ------------------------
#   -----                                      -----   #
fps = 10
size = (2560, 1920)
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
#outVideo = cv2.VideoWriter('saveDir.avi',fourcc,fps,size)
temp_image  = cv2.imread('./position_init/Oth_image/fisheye_catch_org.jpg')

p5s_targetObj_x = []
p5s_targetObj_y = []
# Funtion to update the data and display in the plot
def update():

    dataOk = 0
    global targetObj
    global pointObj
    global tStart
    x = []
    y = []

    # Read and parse the received data
    #dataOk, frameNumber, detObj = readAndParseData16xx(Dataport, configParameters)
    time.sleep(0.02)
    dataOk, frameNumber, targetObj, pointObj = readAndParseData16xx(Dataport, configParameters)

    if dataOk:
        #print(detObj)
        #print(targetObj)
        x = -targetObj["posX"]
        y = targetObj["posY"]

        '''print("x: " + str(x))
        print("y: " + str(y))
        for i in range(len(x)):
            p5s_targetObj_x.append(x[i])
            p5s_targetObj_y.append(y[i])
        print("p5s_targetObj_x: " + str(p5s_targetObj_x))
        print("p5s_targetObj_y: " + str(p5s_targetObj_y))

        tEnd_p5s = time.time()#計時結束
        if((tEnd_p5s - tStart_p5s) >= 0.1):
            p5s_targetObj_x = []
            p5s_targetObj_y = []
            tStart_p5s = time.time()'''

        global temp_image

        tEnd = time.time()
        if ((tEnd - tStart) >= 0.6):
            tStart = time.time()
            image = ipcam.getframe()
            temp_image = image
        else:
            image = temp_image

        '''cv2.ellipse(image, (1280, 960), (long_axis_in_fisheye[0], short_axis_in_fisheye[0]), 0, 0, 360, (0, 255, 0), 4)
        cv2.ellipse(image, (1280, 960), (long_axis_in_fisheye[1], short_axis_in_fisheye[1]), 0, 0, 360, (0, 255, 0), 4)
        cv2.ellipse(image, (1280, 960), (long_axis_in_fisheye[2], short_axis_in_fisheye[2]), 0, 0, 360, (0, 255, 0), 4)
        cv2.ellipse(image, (1280, 960), (long_axis_in_fisheye[3], short_axis_in_fisheye[3]), 0, 0, 360, (18, 153, 255), 4)
        cv2.ellipse(image, (1280, 960), (long_axis_in_fisheye[4], short_axis_in_fisheye[4]), 0, 0, 360, (0, 255, 0), 4)'''

        ''''if((tEnd - tStart) >= 5.0):
            tStart = time.time()
            image = cv2.imread('./position_init/Oth_image/fisheye_catch_org.jpg')
            cv2.ellipse(image, (1280, 960), (395, 329), 0, 0, 360, (0, 255, 0), 4)
            cv2.ellipse(image, (1280, 960), (565, 471), 0, 0, 360, (0, 255, 0), 4)
            cv2.ellipse(image, (1280, 960), (671, 559), 0, 0, 360, (0, 255, 0), 4)
            cv2.ellipse(image, (1280, 960), (737, 614), 0, 0, 360, (0, 255, 0), 4)
            cv2.ellipse(image, (1280, 960), (793, 661), 0, 0, 360, (0, 255, 0), 4)
            temp_image = image
        else:
            image = temp_image'''

        for numOfDetect in range(len(x)):
            draw_coor_x_pixel_contrast = 0.0
            draw_coor_y_pixel = 0.0
            x[numOfDetect] = x[numOfDetect] *100
            y[numOfDetect] = y[numOfDetect] *100
            d_x1, d_y1 = draw_coor(x[numOfDetect], y[numOfDetect])
            draw_coor_x_pixel, draw_coor_y_pixel = coordinate_theta(d_x1, d_y1)
            draw_coor_x_pixel_contrast = x_pixel_contrast(draw_coor_x_pixel)
            draw_coor_y_pixel = round(draw_coor_y_pixel, 4)
            draw_coor_x_pixel_contrast = round(draw_coor_x_pixel_contrast, 4)
            print(str(numOfDetect) + ": " + str(draw_coor_x_pixel_contrast) + ", " + str(draw_coor_y_pixel))

            cv2.circle(image,(int(draw_coor_x_pixel_contrast), int(draw_coor_y_pixel)), 15, (0, 0, 255), -1)

            del draw_coor_x_pixel_contrast
            del draw_coor_y_pixel

        #print(" ")
        #image = cv2.resize(image, (960, 720), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)


        # 顯示圖片
        cv2.imshow('My Image', image)
        #outVideo.write(image)
        cv2.moveWindow('My Image', 0, 0)
        # 按下任意鍵則關閉所有視窗
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()


    QtGui.QApplication.processEvents()

    return dataOk


# -------------------------    MAIN   -----------------------------------------

# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)


# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# 暫停1秒，確保影像已經填充
time.sleep(1)
# Main loop
#detObj = {}
targetObj = {}
pointObj = {}
frameData = {}
currentIndex = 0

'''cap = cv2.VideoCapture("rtsp://140.124.182.158/video2")
ret,image = cap.read()
if ret:
    cv2.imshow("image",image)
    cv2.waitKey(0)'''

while True:
    try:
        # Update the data and check if the data is okay
        dataOk = update()

        if dataOk:
            # Store the current frame into frameData
            #print('DataOK and go to save it')
            #frameData[currentIndex] = detObj
            frameData[currentIndex] = targetObj
            currentIndex += 1

        #time.sleep(0.033) # Sampling frequency of 30 Hz
        time.sleep(0.03) # Sampling frequency of 20 Hz

    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        #CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        #outVideo.release()
        cv2.destroyAllWindows()
        ipcam.stop()
        break
