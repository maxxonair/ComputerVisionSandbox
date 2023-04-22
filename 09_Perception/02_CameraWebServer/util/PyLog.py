# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:57:19 2022

Class to add simplified logging to Breadboards

@author: braun_m
"""
import os 
from datetime import datetime

class PyLog:
    
    # Flag: Enable console prints 
    flagIsConsolePrint = False 
    # Flag: Create and save logging text file
    flagIsSaveLogFile  = False 
    # Logging file path 
    strLogFilePath = ""
    
    iCountWarning   = 0
    iCountError     = 0
    iCountLogMsgs   = 0
    
    # Create message when initializing logging
    def __pLogInitMsg( self ):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        strLogFinal = "[INI] Logging started: " + dt_string
        # Write to console 
        if self.flagIsConsolePrint:
            print(strLogFinal)
            print()
        # Write Log Message to file 
        if self.flagIsSaveLogFile:
            self.fileOutLog.write(strLogFinal+"\n")
        
    # Create message to close-out logging
    def __pLogCloseMsg( self ):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        strLogFinal = "[END] Logging completed: " + dt_string
        strStats = f"\n[---] {self.iCountLogMsgs} logged messages.\n[WRN] {self.iCountWarning} Warnings.\n[ERR] {self.iCountError} Errors. \n"
        # Write to console 
        if self.flagIsConsolePrint:
            print()
            print(strLogFinal)
            print(strStats)
        # Write Log Message to file 
        if self.flagIsSaveLogFile:
            self.fileOutLog.write("\n\n"+strLogFinal)
            self.fileOutLog.write("\n"+strStats)
    
    def __createLineNumber(self):
        if self.iCountLogMsgs < 10:
            return "[000"+str(self.iCountLogMsgs)+"]"
        elif self.iCountLogMsgs < 100:
            return "[00"+str(self.iCountLogMsgs)+"]"
        elif self.iCountLogMsgs < 1000:
            return "[0"+str(self.iCountLogMsgs)+"]"
        else:
            return "["+str(self.iCountLogMsgs)+"]"
        
    # Initialize logging instance 
    def __init__( self, sFolderPath='', strLogFileName='', flagIsConsolePrint=True, flagIsSaveLogFile=False):
        
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        
        # Assign log file path
        self.strLogFilePath = sFolderPath + '/' + dt_string + "__" + strLogFileName + ".txt"
        self.flagIsConsolePrint = flagIsConsolePrint
        self.flagIsSaveLogFile  = flagIsSaveLogFile
        
        if self.flagIsSaveLogFile:
            # Delete log file if exists
            try:
                os.remove(self.strLogFilePath)
            except OSError:
                pass
            
            # Init new log file in append mode 
            self.fileOutLog = open(self.strLogFilePath, "a")
        
        # Add first log line
        self.__pLogInitMsg()
        
    # Close logging 
    def close(self):
        self.__pLogCloseMsg()
        if self.flagIsSaveLogFile:
            self.fileOutLog.close()
        
    # Append (neutral) Message     
    def pLogMsg( self, strLogMsg ):
        # Update Counter 
        self.iCountLogMsgs = self.iCountLogMsgs + 1
        strLogFinal = f'{self.__createLineNumber()}[MSG] {strLogMsg}'
        # Write to console 
        if self.flagIsConsolePrint:
            print(strLogFinal)
        # Write Log Message to file 
        if self.flagIsSaveLogFile:
            self.fileOutLog.write("\n"+strLogFinal)
        
    # Append Error
    def pLogErr( self, strLogMsg ):
        # Update Counter
        self.iCountError = self.iCountError + 1
        self.iCountLogMsgs = self.iCountLogMsgs + 1
        strLogFinal = f'self.__createLineNumber()[ERR] {strLogMsg}'
        # Write to console 
        if self.flagIsConsolePrint:
            print(strLogFinal)
        # Write Log Message to file 
        if self.flagIsSaveLogFile:
            self.fileOutLog.write("\n"+strLogFinal)
    
    # Append Warning
    def pLogWrn( self, strLogMsg ):
        # Update Counter
        self.iCountWarning = self.iCountWarning + 1
        self.iCountLogMsgs = self.iCountLogMsgs + 1
        strLogFinal = f'self.__createLineNumber()[WRN] {strLogMsg}'
        # Write to console 
        if self.flagIsConsolePrint:
            print(strLogFinal)
        # Write Log Message to file 
        if self.flagIsSaveLogFile:
            self.fileOutLog.write("\n"+strLogFinal)
    