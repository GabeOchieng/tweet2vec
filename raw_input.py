'''
raw_input.py
Module for processing raw_input into the associated character and word embedding matrices for input into the neural network architecture
'''

import string 
import numpy as np 
#np.set_printoptions(threshold=np.nan)

class RawInput: 
	def __init__(self,input_txt,char_options=None,type_matrix='char',max_char=140,max_words=50): 

		# set up the possible characters allowed for the character matrix embedding, a base set of ascii_lower, numbers, and punctuation is included. Adding input char_options extends this base set.
		if char_options == None: 
			self.char_options = string.ascii_lowercase+string.digits+string.punctuation
		else: 
			self.char_options = string.ascii_lowercase+string.digits+string.punctuation
			for c in char_options: 
				if c not in self.char_options: 
					self.char_options+=c

		self.raw = input_txt
		self.max_char = max_char 		#maximum characters allowed for an input 
		self.max_words = max_words		#maximum words allowed for an input
		self.n_valid_chars = len(self.char_options)		#number of valid characters
		self.char_matrix = self.getCharMatrix(type_matrix=type_matrix)

	def getCharMatrix(self,type_matrix='char'): 
		'''
		Purpose: 
		Computes the input to the character-based embedding Neural Net. The character embedding can be done either as a one-hot encoding for each character in the input ('char') or as a sum of one-hot encodings for each character for each word (separated by spaces) in the input ('word'). 

		inputs: 
		type_matrix (str) - 
			'char' = 1-hot encoding for a single character per row
			'word' = sum of 1-hot encoded characters per row 
		'''
		if type_matrix == 'char': 
			C = np.zeros((self.max_char,self.n_valid_chars))		# max_char by n_valid_chars
			for i,c in enumerate(self.raw):
				if i >= self.max_char: 
					break
				try: 
					c_pos = self.char_options.index(c)
					C[i,c_pos] = 1 
				except ValueError: 
					pass  

			return C 

		elif type_matrix == 'word': 
			C = np.zeros((self.max_words,self.n_valid_chars))		# max_words by n_valid_chars
			for i,word in enumerate(self.raw.split(' ')):
				if i >= self.max_words: 
					break
				for c in word: 
					try: 
						c_pos = self.char_options.index(c)
						C[i,c_pos] += 1
					except ValueError: 
						pass 
			return C 


