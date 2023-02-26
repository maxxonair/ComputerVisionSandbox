import serial

def printImuMsgs():
  ser = serial.Serial()

  # Set Baud rate. Default for all projects 115200
  ser.baurate = 115200

  ser.port = ''

  ser.open()

  total = 0
  while total < len(values):
    print(ord(ser.read(1)))
    total = total + 1

  ser.close()