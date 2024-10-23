# -*-coding:Latin-1 -*

import os      # Management of screen input/output
import pickle  # Manage the savings of our local objects into a file
import csv     # To read csv file
import sys     # To send command line info
import time    # To measure time performance

from collections import defaultdict # To manage a dictionary of list
from datetime import datetime, timedelta
import BOM

##############################################################################################
#
# Writer : Generic class to write csv files
#
#   Main attributes:
#      - filename			: name of the output file that we will generate
#      - BOM				: our Business Object Manager that will store what we read
#      - debugLevel			: define the trace level that we want to use
#      - debugLevel			: define the trace level that we want to use
#
###############################################################################################
class CsvFileWriter:
	""" Generic class """
	def __init__(self, filename, BOM, debugLevel_p = 0):
		## Constructor
		self._filename		= filename
		self._BOM			= BOM
		self._debugLevel	= debugLevel_p
		self._delimiter		 = ','

	def __str__(self):
		## Printor
		return
	def write(self):
		## Read function
		return
	
	""" ========= GETTERS and SETTERS =========== """
	""" Filenalme """
	def _get_filename(self):
		return self._filename
	def _set_filename(self, filename_l):
		self._filename = filename_l
	filename = property(_get_filename, _set_filename)
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
	""" Delimiter used to write the file """
	def _get_delimiter(self):
		return self._delimiter
	def _set_delimiter(self, delimiter_l):
		self._delimiter = delimiter_l
	delimiter = property(_get_delimiter, _set_delimiter)

##############################################################################################
#
# Results : Inheritance from class Writer, will write the output of our work from our BOM
#
#   Main attributes:
#
#
###############################################################################################
class Results(CsvFileWriter):
	def __init__(self, filename, BOM):
		""" Inheritance from CsvFileWriter """
		CsvFileWriter.__init__(self, filename, BOM)
	
	def __str__(self):
		## Printor
		return
	
	def write_all(self):
		start = time.time()
		# local parameters
		
		with open(self._filename, 'wb') as csvFile:
			writer_l = csv.writer(csvFile)
			
			# write hearders
			writer_l.writerow(["FC","SOG","Carrier","Sort_Code","CPT_Day","CPT_Hour","zip","weightBand","cps","ead","TT","NbPackages"])

			# write the rest
			for FC_l in self.BOM.listFCWithDemand:
				for CPT_l in FC_l.listOfCPTs:
					for CPT_sort_code_l in CPT_l.listOfCPT_sortCode:
						sort_code_l = CPT_sort_code_l.sort_code
						for CPT_zip_wb in CPT_sort_code_l.listOfCPT_zip_wb:
							if( CPT_zip_wb.nbPackages <= 0 ):
								continue
							CPT_datetime_l	= str(CPT_l.datetime+timedelta(hours=CPT_l.hour, minutes=CPT_l.minute))
							sog_l			= CPT_zip_wb.SO
							writer_l.writerow(["{}".format(FC_l.name),
							"{}".format(sog_l),
							"{}".format(CPT_l.carrier),
							"{}".format(sort_code_l),
							"{}".format(CPT_datetime_l[:10]),
							"{}".format(CPT_datetime_l[11:16]),
							"{}".format(CPT_zip_wb.zipCode),
							"{}".format(CPT_zip_wb.weightBand),
							"{:.2f}".format(CPT_zip_wb.cps),
							"{}".format(str(CPT_zip_wb.ead)[:10]),
							"{}".format(CPT_zip_wb.TT),
							"{}".format(CPT_zip_wb.nbPackages)])
							# print ["{}".format(FC_l.name),
							# "{}".format(sog_l),
							# "{}".format(CPT_l.carrier),
							# "{}".format(sort_code_l),
							# "{}".format(CPT_datetime_l[:10]),
							# "{}".format(CPT_datetime_l[11:16]),
							# "{}".format(CPT_zip_wb.zipCode),
							# "{:.2f}".format(CPT_zip_wb.cps),
							# "{}".format(str(CPT_zip_wb.ead)[:10]),
							# "{}".format(CPT_zip_wb.TT),
							# "{}".format(CPT_zip_wb.nbPackages)]
								
		end = time.time()
		print("[WRITE],      x Elapsed time: {}".format(end - start))
	
	def write_zip2(self):
		start = time.time()
		# local parameters
		
		dictNbPackages_l	= {}
		dictCPS_l			= {}
		dictTT_l			= {}
		dictEAD_l			= {}
		dictNbZip5_l		= {}
		with open(self._filename, 'wb') as csvFile:
			writer_l = csv.writer(csvFile)
			
			# write hearders
			writer_l.writerow(["FC","SOG","Carrier","Sort_Code","CPT_Day","CPT_Hour","zip","cps","ead","TT","NbPackages"])

			# aggregate zip2
			for FC_l in self.BOM.listFCWithDemand:
				for CPT_l in FC_l.listOfCPTs:
					for CPT_sort_code_l in CPT_l.listOfCPT_sortCode:
						sort_code_l = CPT_sort_code_l.sort_code
						for CPT_zip_wb in CPT_sort_code_l.listOfCPT_zip_wb:
							if( CPT_zip_wb.nbPackages <= 0 ):
								continue
							CPT_datetime_l	= str(CPT_l.datetime+timedelta(hours=CPT_l.hour, minutes=CPT_l.minute))
							sog_l			= CPT_zip_wb.SO
							tuple_l = (FC_l.name, sog_l, str(CPT_l.carrier), sort_code_l, CPT_l, CPT_zip_wb.zipCode[0:2])
							
							if( tuple_l not in dictNbPackages_l.keys() ):
								dictNbPackages_l[tuple_l]	= 0
								dictCPS_l[tuple_l]			= 0
								dictTT_l[tuple_l]			= 0
								dictEAD_l[tuple_l]			= 0
								dictNbZip5_l[tuple_l]		= 0
							
							dictNbPackages_l[tuple_l]	+= CPT_zip_wb.nbPackages
							dictCPS_l[tuple_l]			+= CPT_zip_wb.nbPackages * CPT_zip_wb.cps
							dictTT_l[tuple_l]			+= CPT_zip_wb.nbPackages * CPT_zip_wb.TT
							dictEAD_l[tuple_l]			+= CPT_zip_wb.nbPackages * (CPT_zip_wb.ead - CPT_l.datetime).total_seconds()
							dictNbZip5_l[tuple_l]		+= 1
			
			# write the file
			for tuple_l in dictNbPackages_l.keys():
				FC_name_l		= tuple_l[0]
				sog_l			= tuple_l[1]
				carrier_name_l	= tuple_l[2]
				sort_code_l		= tuple_l[3]
				CPT_l			= tuple_l[4]
				CPT_datetime_l	= str(CPT_l.datetime+timedelta(hours=CPT_l.hour, minutes=CPT_l.minute))
				zip2_l			= tuple_l[5]
				nbPackages_l	= dictNbPackages_l[tuple_l]
				cps_l			= dictCPS_l[tuple_l] / float(nbPackages_l)
				TT_l			= dictTT_l[tuple_l] / float(nbPackages_l)
				ead_l			= CPT_l.datetime + timedelta(seconds=int(dictEAD_l[tuple_l]/float(nbPackages_l)))
							
				writer_l.writerow(["{}".format(FC_name_l),
				"{}".format(sog_l),
				"{}".format(carrier_name_l),
				"{}".format(sort_code_l),
				"{}".format(CPT_datetime_l[:10]),
				"{}".format(CPT_datetime_l[11:16]),
				"{}".format(zip2_l),
				"{:.2f}".format(cps_l),
				"{}".format(str(ead_l)[:10]),
				"{}".format(TT_l),
				"{}".format(nbPackages_l)])		
								
		end = time.time()
		print("[WRITE],      x Elapsed time: {}".format(end - start))
	
	def write(self):
		self.write_all()
		return
