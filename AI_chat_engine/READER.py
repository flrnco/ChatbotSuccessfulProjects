# -*-coding:Latin-1 -*

import os        # Management of screen input/output
import math        # All math functions
import re        # Regular Expressions
import random    # Random functions
import pickle    # Manage the savings of our local objects into a file
import csv        # To read csv file
import sys        # To send command line info
import time        # To measure time performance
#import xlrd        # to read excel documents
import numpy as np    # to vectorize some functions
from datetime import datetime, timedelta

from collections import defaultdict # To manage a dictionary of list
from AI_chat_engine import BOM

dictWeekDay_g = {
        "All": 10,
        "Sun": 1,
        "Mon": 2,
        "Tue": 3,
        "Wed": 4,
        "Thu": 5,
        "Fri": 6,
        "Sat": 7,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7
    }

##############################################################################################
#
# Reader : Generic class to read csv files
#
#   Main attributes:
#      - BOM                : our Business Object Manager that will store what we read
#      - nbErrors            : nb errors observed in input file, we stop the execution if > 0
#      - debugLevel            : define the trace level that we want to use
#
###############################################################################################
class Reader:
    """ Generic csv file reader """
    def __init__(self, filename, BOM, debugLevel = 0):
        ## Constructor
        self._BOM            = BOM
        self._nbErrors        = 0
        self._debugLevel    = debugLevel

    def __str__(self):
        ## Printor
        return
        
    """ ========= GETTERS and SETTERS =========== """
    """ Business Objects Manager """
    def _get_BOM(self):
        return self._BOM
    def _set_BOM(self, BOM_l):
        self._BOM = BOM_l
    BOM = property(_get_BOM, _set_BOM)
    """ Debug level manager: 0 = no trace, 1 = debug trace active """
    def _get_debugLevel(self):
        return self._debugLevel
    def _set_debugLevel(self, debugLevel_l):
        self._debugLevel = debugLevel_l
    debugLevel = property(_get_debugLevel, _set_debugLevel)
    
    """ ========= CLASS FUNCTIONS =========== """
    def manageErrors(self):
        # Error management
        if self._nbErrors > 0:
            print("!! READ ERROR: You have {} error(s) described above to correct".format(self._nbErrors))
            sys.exit()
        return
    
    def readAllInputFiles():
        return
    
    def read(self):
        ## Read function: !! DON'T SURCHARGE THIS FUNCTION !!
        ## => you need to surcharge 'readAllInputFiles' instead to read your files
        self.readAllInputFiles()
        self.manageErrors()
        return
        
    def check_Zip_Code_Format(self, zipCode_l, worksheet_name, idxRow):
        # ERROR messages
        if ( zipCode_l != round(zipCode_l,0) or zipCode_l <= 0 ):
            print("READ ERROR: {} is not a valid zip code on line {} of sheet {}".format(zipCode_l, idxRow+1, worksheet_name))
            print(" -- Expected: Integer >= 0")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_StrictPositiveInt_Format(self, value, worksheet_name, idxRow):
        # ERROR messages
        if not isinstance( value, int ) or value <= 0:
            print("READ ERROR: {} is not a valid value on line {} of sheet {}".format(value, idxRow+1, worksheet_name))
            print(" -- Expected: Integer > 0")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_PositiveInt_Format(self, value, worksheet_name, idxRow):
        # ERROR messages
        if not isinstance( value, int ) or value < 0:
            print("READ ERROR: {} is not a valid value on line {} of sheet {}".format(value, idxRow+1, worksheet_name))
            print(" -- Expected: Integer >= 0")
            self._nbErrors += 1
            return -1
        return 0

    def check_Percentage_Format(self, value, worksheet_name, idxRow):
        # ERROR messages
        if not isinstance( value, float ) or value < 0 or value > 1:
            print("READ ERROR: {} is not a valid value on line {} of sheet {}".format(value, idxRow+1, worksheet_name))
            print(" -- Expected: percentage between 0 and 1")
            self._nbErrors += 1
            return -1
        return 0

    def check_CarrierName(self, carrier, worksheet_name, idxRow):
        # ERROR messages
        if carrier not in self._BOM.list_CarriersWithoutAMZL:
            print("READ ERROR: {} is not a valid carrier name on line {} of sheet {}".format(carrier, idxRow+1, worksheet_name))
            print(" -- Expected: carrier name existing in the tab CarrierMix")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_ZipCode(self, zipCode, worksheet_name, idxRow):
        # ERROR messages
        if zipCode not in self._BOM.list_ZipCodes:
            print("READ ERROR: {} is not a valid zip code on line {} of sheet {}".format(zipCode, idxRow+1, worksheet_name))
            print(" -- Expected: zip code existing in the tab Mapping_Zip_Codes")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_ProvinceName(self, province, worksheet_name, idxRow):
        # ERROR messages
        if province not in self._BOM.list_Provinces:
            print("READ ERROR: {} is not a valid province on line {} of sheet {}".format(province, idxRow+1, worksheet_name))
            print(" -- Expected: province name existing in the tab Mapping_Zip_Codes")
            self._nbErrors += 1
            return -1
        return 0
            
    def check_DayInMonth(self, day, worksheet_name, idxRow):
        # ERROR messages
        if day <=0 or day>31:
            print("READ ERROR: {} is not a valid day in the month on line {} of sheet {}".format(day, idxRow+1, worksheet_name))
            print(" -- Expected: day should be strictly higher than 0 and strictly lower than 32")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_Month(self, month, worksheet_name, idxRow):
        # ERROR messages
        if month <=0 or month>12:
            print("READ ERROR: {} is not a valid month on line {} of sheet {}".format(month, idxRow+1, worksheet_name))
            print(" -- Expected: month should be strictly higher than 0 and strictly lower than 13")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_Year(self, year, worksheet_name, idxRow):
        # ERROR messages
        if year <=2000:
            print("READ ERROR: {} is not a valid year on line {} of sheet {}".format(year, idxRow+1, worksheet_name))
            print(" -- Expected: year should be higher than 2000")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_Hour(self, hour, worksheet_name, idxRow):
        # ERROR messages
        if hour < 0 or hour > 23:
            print("READ ERROR: {} is not a valid hour on line {} of sheet {}".format(hour, idxRow+1, worksheet_name))
            print(" -- Expected: an hour should be an integer between 0 and 23")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_Minute(self, minute, worksheet_name, idxRow):
        # ERROR messages
        if minute < 0 or minute > 59:
            print("READ ERROR: {} is not a valid minute on line {} of sheet {}".format(minute, idxRow+1, worksheet_name))
            print(" -- Expected: a minute should be an integer between 0 and 59")
            self._nbErrors += 1
            return -1
        return 0
    
    def check_elementInList(self, element, dictWhereElementIsAKey, typeOfElement, filename, idxRow, filename2):
        # ERROR messages
        if element not in dictWhereElementIsAKey:
            print("READ ERROR: {} is not a valid {} on line {} of file {}".format(element, typeOfElement, idxRow, filename))
            print(" -- Expected: this {} should already be defined in {}".format(typeOfElement, filename2))
            self._nbErrors += 1
            return -1
        return 0

## Support functions
def reformat_date(dateFormatYYYYMMDD_p):
    yyyy_l    = int(dateFormatYYYYMMDD_p[0:4])
    mm_l    = int(dateFormatYYYYMMDD_p[4:6])
    dd_l    = int(dateFormatYYYYMMDD_p[6:8])
    return datetime(yyyy_l,mm_l,dd_l)

def stringToWeekDay(strWeekDay_p):
    if strWeekDay_p not in dictWeekDay_g.keys():
        return -1
    return dictWeekDay_g.get(strWeekDay_p)
        
##############################################################################################
#
# Parameters :    Inheritance from class Reader, will read all the input file and store data
#                in our Business Object Manager
#
#   Main attributes:
#      - names of all the files that we will read
#
###############################################################################################
class Parameters(Reader):
    def __init__(self, BOM, mode = BOM.TRAINING_MODE_g, debugLevel = 0):
        """ Inheritance from Reader master class """
        Reader.__init__(self, '', BOM, debugLevel)
        
        self._glove_embeddings                = "input/glove.6B.50d.txt"
        self._trainingSet                   = "input/trainingset_sentences.csv"
        self._training_or_test              = mode # can be BOM.TRAINING_MODE_g or BOM.TEST_MODE_g
       
    
    def __str__(self):
        ## Printor
        return    
                            
    def readAllInputFiles(self):
        
        start_time=time.time()
        self.read_glove_embeddings()
        if self._training_or_test == BOM.TRAINING_MODE_g:
            self.read_trainingSet()
        
        return
    
    def read_trainingSet(self):
        """ Read our training set file """
        if self.debugLevel>0:
            print("[READER],   GloVe Word Embedding...")
        
        idxRow_l    = 1
        try:
            # Read the CSV file and store data from each column into corresponding lists
            with open(self._trainingSet, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                
                for row in reader:
                    self.BOM.training_sentences.append(row['Sentences'])  # Store the sentence
                    self.BOM.questions.append(int(row['Question']))       # Store the question (convert to int)
                    self.BOM.subjects.append(row['Subject'])              # Store the subject
                    self.BOM.action_verbs.append(row['Action_verb'])       # Store the action verb
                    self.BOM.objectives.append(int(row['Objective']))      # Store the objective (convert to int)
        
        except IndexError:
            print("[READER],   Index error on line {} of the file {}...".format(idxRow_l, self._trainingSet))
            sys.exit()
        
        ## Test that we read and store everything properly
        #idx_l = 0
        #print("test reading training set")
        #for sentence_l in self.BOM.training_sentences:
        #    print("Sentence :",sentence_l)
        #    print("--question :",self.BOM.questions[idx_l])
        #    print("--objective :",self.BOM.objectives[idx_l])
        #    print("--subjects :",self.BOM.subjects[idx_l])
        #    print("--action_verbs :",self.BOM.action_verbs[idx_l])
        #    idx_l += 1
        
        return
    
    def read_glove_embeddings(self):
        """ Read our word embeddings file """
        if self.debugLevel>0:
            print("[READER],   GloVe Word Embedding...")
        
        idxRow_l    = 1
        try:
            with open(self._glove_embeddings, 'r', encoding='utf-8', errors='replace') as f:
                
                for line in f:
                    #print(line)
                    line = line.strip().split()
                    curr_word = line[0]
                    self.BOM.words_in_vocabulary.add(curr_word)
                    self.BOM.word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            

        except IndexError:
            print("[READER],   Index error on line {} of the file {}...".format(idxRow_l, self._file_zipCodeShare))
            sys.exit()
        
        #for FC_l in self.BOM.zipShareBy_FC_Zip.keys():
        #    for zip_l in self.BOM.zipShareBy_FC_Zip[FC_l]:
        #        print("{},{},{}".format(FC_l,zip_l, self.BOM.zipShareBy_FC_Zip[FC_l][zip_l]))
        #sys.exit()
        return
    