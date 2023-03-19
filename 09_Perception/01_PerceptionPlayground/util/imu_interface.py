

import serial
import time 
import sys 
import glob
import numpy as np
import math

class ImuConnector():

    # Unique device identifier for IMU6050 StereoBench IMU
    USB_IMU_DEVICE_ID = 605045772

    # Number of packets in IMU message
    NUM_IMU_MEAS_PACKETS = 9

    # IMU message delimiter
    messageDelimiter = ','

    baud_rate = 115200 
    usb_port = ''
    serial_timeout_s=1


    # Number of messages to wait for until parsing and evaluation, while looking for the 
    # correct imu port 
    port_finder_msg_attempts = 5

    def __init__(self):
        # Find usb ports
        usb_ports = self._listSerialPorts()
        if len(usb_ports) > 1:
            print(f"Multiple serial ports ({len(usb_ports)}) found. Searching for IMU ... ")
            print(f'Detected ports: {usb_ports}')
            isValidPortFound = False
            for portCounter, usb_port in enumerate(usb_ports):
                print(f'Open and check port {portCounter}')
                # Try to open and read from port 
                self.usb_port = usb_port
                self.serCon = serial.Serial(self.usb_port, 
                                            self.baud_rate, 
                                            timeout=self.serial_timeout_s)

                if self.serCon.isOpen():
                    self.serCon.close()

                # Open connection    
                self._openConnection()

                # Take several messages to make sure a vaild one comes through
                for iCounter in range(self.port_finder_msg_attempts):
                    message = self.serCon.readline()\
                    # If message is completely empty
                    # -> Thats not our port, moving on
                    if message == b'':
                        break

                if (self._parseImuMessage(message))['dev_id'] == self.USB_IMU_DEVICE_ID:
                    self._closeConnection()
                    print(f'IMU found on port {portCounter}')
                    isValidPortFound = True
                    break
            if not isValidPortFound:
                print(f'ERROR: IMU port not found! ')
                print('Exiting')
                exit(1)
        else:
            self.usb_port = usb_ports[0]
            self.serCon = serial.Serial(self.usb_port, self.baud_rate)

            if self.serCon.isOpen():
                self.serCon.close()

    def stream(self):
        self._openConnection()
        
        while True:
            message = self.serCon.readline()
            try:
                measurement = self._parseImuMessage(message)
                print(f'[{measurement["msg_id"]}|{measurement["msg_id"]}] == {measurement["temp_degC"]} {measurement["ax_mss"]} {measurement["ay_mss"]} {measurement["az_mss"]}')
            except:
                print('Error: Parsing message failed')
                
            time.sleep(0.1)
        
    def takePoseMeasurement(self, numMeasurements):
        self._openConnection()
        prevID = -1
        measCounter = 0
        while measCounter <= numMeasurements:
            message = self.serCon.readline()
            try:
                measurement = self._parseImuMessage(message)
                if measurement["msg_id"] != prevID and measurement["msg_id"] != -1:
                    # accVector_mss.append([float(measurement["ax"]), float(measurement["ax"]), float(measurement["ax"])])
                    accMeas = [float(measurement["ax_mss"]), 
                               float(measurement["ay_mss"]), 
                               float(measurement["az_mss"])]
                    tempMeas = float(measurement["temp_degC"])
                    if measCounter == 0:
                        accVector_mss  = accMeas
                        tempVector_deg = tempMeas
                    else:
                        accVector_mss = np.vstack((accVector_mss, accMeas))
                        tempVector_deg = np.vstack((tempVector_deg, tempMeas))
                    measCounter = measCounter + 1
                prevID = measurement["msg_id"]
            except:
                print('Error: Parsing message failed')
            time.sleep(0.1)
            
        accVector_average_mss = accVector_mss.mean(axis=0)
        accVector_stdv_mss    = accVector_mss.std(axis=0)
        temp_average_deg      = tempVector_deg.mean(axis=0)
        self._closeConnection()
        return (accVector_mss, accVector_average_mss, accVector_stdv_mss, temp_average_deg)
    
    #-------------------------------------------------------------------------
    #       [Private Functions]
    #-------------------------------------------------------------------------
    
    def _parseImuMessage(self, message_in):
        message = str(message_in.decode("utf-8") )
        strArray = message.split(self.messageDelimiter)
        if len(strArray) < self.NUM_IMU_MEAS_PACKETS:
            # Message not valid
            measurement = {
                'dev_id': -1,
                'msg_id': -1,
                'temp_degC':  -1,
                'ax_mss': -1,
                'ay_mss': -1,
                'az_mss': -1,
                'gx_rads': -1,
                'gy_rads': -1,
                'gz_rads': -1,
            }
        else:
            # Message not valid -> Parsing to dict
            measurement = {
                'dev_id': int(strArray[0]),
                'msg_id': int(strArray[1]),
                'temp_degC':  float(strArray[2]),
                'ax_mss': float(strArray[3]),
                'ay_mss': float(strArray[4]),
                'az_mss': float(strArray[5]),
                'gx_rads': float(strArray[6]),
                'gy_rads': float(strArray[7]),
                'gz_rads': float(strArray[8]),
                }

        return measurement

    def _openConnection(self):
        if self.serCon.isOpen():
            self.serCon.close()
        
        # Open serial port 
        self.serCon.open()

    def _closeConnection(self):
        print("close serial")
        self.serCon.close()
        
    def _listSerialPorts(self):
        """ Lists serial port names

            :raises EnvironmentError:
                On unsupported or unknown platforms
            :returns:
                A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result
    
    def _computeRollPitch(self, gravityVector):
        accy =   gravityVector[0]
        accz = - gravityVector[1]
        accx =   gravityVector[2]
        pitch_deg = math.degrees(float(
        math.asin( 
                    accx / math.sqrt( float(  accx*accx 
                                            + accy*accy 
                                            + accz*accz))) ))
        
        roll_deg   = math.degrees(float( math.atan2(accy,accz) ))
        
        return roll_deg, pitch_deg

    #-------------------------------------------------------------------------