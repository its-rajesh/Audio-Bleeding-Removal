####################################################################################################
#                       FUNCTIONS TO HANDLE DYNAMIC INPUT SPECTOGRAM SIZE                          #
####################################################################################################
# Author: Rajesh R (Git: its-rajesh)
# Date: 14/Januray/2023

import numpy as np


class DynamicInput:

    def __init__(self):
        pass


    '''
    FUNCTION: ENCODES (SPLITS & MERGE) THE INPUT SPECTOGRAM INTO VARIOUS BINS (See Ref. Figure)
    USE: To handle dynamic inputs to neural netwok.
    Parameters:
    (1) array: Input spectogram of size m*n
    (2) frame: No of bins to be grouped together. Default is 3, If frame = no of columns in input spect/3, 
        and skip=3, then output is same as input.
    (3) skip: Overlap within the bins (1<=skip<=3), default is 1 to maintain loss of info.
        either use skip =1 or frame = 1 for no loss of info.

    Note: Batches/Grouping is always 3.
    '''

    def encode(self, array, frame=3, skip=1):
        
        data = []
        for i in range(0, array.shape[1]-(3*frame-1), skip):
            y_ = array[:, i:i+frame]
            y = array[:, i+frame:i+(2*frame)]
            y__ = array[:, i+(2*frame):i+(3*frame)]
            concat = np.concatenate((y_, y, y__), axis=0, casting="same_kind")
            data.append(concat)
            
        self.encoded = np.array(data)
        return self.encoded


    '''
    FUNCTION: DECODES (REVERSE OF ENCODE) THE SPLIT BINS INTO ORIGINAL SPECTOGRAM (See Ref. Figure)
    USE: To handle dynamic inputs to neural netwok.
    Parameters:
    (1) array: Input spectogram bins of size l*m*n
    (2) frame: Same value as given in encode function
    (3) skip: Same value as given in encode function

    Note: Batches/Grouping is always 3.
    '''

    def decode(self, array, frame=3, skip=1):
        reconst = []
        l, m, n = array.shape
        flag = False
        for i in array:
            
            y_ = i[:int(m/3)]
            y = i[int(m/3):int(2*(m/3))]
            y__ = i[int(2*(m/3)):m]

            if skip == 1:
                if flag:
                    if frame >= 3:
                        t = np.array([y_[:, 2]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                    if frame == 1:
                        t = np.array([y_[:, 0]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                    if frame == 2:
                        t = np.array([y_[:, 1]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                else:
                    reconst = y_
                    
            if skip == 2:
                if flag:
                    if frame >=3:
                        t = np.array([y_[:, 1:2]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                    if frame == 1:
                        t = np.array([y_[:, 0]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                    if frame == 2:
                        t = np.array([y_[:, 1]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                else:
                    reconst = y_
                    
            if skip == 3:
                if flag:
                    if frame>=3:
                        reconst = np.concatenate((reconst, y_), axis=1, casting="same_kind")
                    if frame == 1:
                        t = np.array([y_[:, 0]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                    if frame == 2:
                        t = np.array([y_[:, 1]]).T
                        reconst = np.concatenate((reconst, t), axis=1, casting="same_kind")
                else:
                    reconst = y_
                    
            flag = True
            
        if skip == 1:
            reconst = np.concatenate((reconst, y, y__), axis=1, casting="same_kind")

        self.decoded = np.array(reconst)
        return self.decoded
        
        
        
    '''
	FUNCTION: ENCODES (SPLITS & MERGE) THE INPUT SPECTOGRAM INTO VARIOUS BINS (See Ref. Figure)
	USE: To handle dynamic inputs to neural netwok.
	Parameters:
	(1) array: Input complex spectogram of size m*n
	(2) frame: No of bins to be grouped together. Default is 3, If frame = no of columns in input spect/3, 
	    and skip=3, then output is same as input.
	(3) skip: Overlap within the bins (1<=skip<=3), default is 1 to maintain loss of info.
	    either use skip =1 or frame = 1 for no loss of info.

	Note: Batches/Grouping is always 3.
	'''
    def encode3D(array, frame=3, skip=1):
	    
        data = []
        for i in range(0, array.shape[1]-(3*frame-1), skip):
            y_ = array[:, i:i+frame]
            y = array[:, i+frame:i+(2*frame)]
            y__ = array[:, i+(2*frame):i+(3*frame)]
            concat = np.concatenate((y_, y, y__), axis=0, casting="same_kind")
            mag = np.abs(concat)
            phase = np.angle(concat)
            data.append(np.stack((mag, phase)))
		
        return np.array(data)
		    
    '''
	FUNCTION: DECODES (REVERSE OF ENCODE) THE SPLIT BINS INTO ORIGINAL SPECTOGRAM (See Ref. Figure)
	USE: To handle dynamic inputs to neural netwok.
	Parameters:
	(1) array: Input complex spectogram bins of size l*2*m*n
	(2) frame: Same value as given in encode function
	(3) skip: Same value as given in encode function

	Note: Batches/Grouping is always 3.
	'''

    def decode3D(array, frame=3, skip=1):
        mag_reconst = []
        l, _, m, n = array.shape
        flag = False
        for i in array:
		
            y_ = i[0][:int(m/3)]
            y = i[0][int(m/3):int(2*(m/3))]
            y__ = i[0][int(2*(m/3)):m]
		
            yp_ = i[1][:int(m/3)]
            yp = i[1][int(m/3):int(2*(m/3))]
            yp__ = i[1][int(2*(m/3)):m]

        if skip == 1:
            if flag:
                t = np.array([y_[:, 2]]).T
                mag_reconst = np.concatenate((mag_reconst, t), axis=1, casting="same_kind")
		        
                tp = np.array([yp_[:, 2]]).T
                phase_reconst = np.concatenate((phase_reconst, tp), axis=1, casting="same_kind")
		        
            else:
                mag_reconst = y_
                phase_reconst = yp_
		        
        if skip == 2:
            if flag:
                t = np.array([y_[:, 1:2]]).T
                mag_reconst = np.concatenate((mag_reconst, t), axis=1, casting="same_kind")
		        
                tp = np.array([yp_[:, 1:2]]).T
                phase_reconst = np.concatenate((phase_reconst, tp), axis=1, casting="same_kind")
		        
            else:
                mag_reconst = y_
                phase_reconst = yp_
		        
        if skip == 3:
            if flag:
                mag_reconst = np.concatenate((mag_reconst, y_), axis=1, casting="same_kind")
                phase_reconst = np.concatenate((phase_reconst, yp_), axis=1, casting="same_kind")
            else:
                mag_reconst = y_
                phase_reconst = yp_
		        
        flag = True
		
        if skip == 1:
            mag_reconst = np.concatenate((mag_reconst, y, y__), axis=1, casting="same_kind")
            phase_reconst = np.concatenate((phase_reconst, yp, yp__), axis=1, casting="same_kind")

        return np.stack((mag_reconst, phase_reconst))

