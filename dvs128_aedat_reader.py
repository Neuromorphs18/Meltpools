# Import the required libraries to load and execute the code
import matplotlib
import numpy as np
import pdb
import matplotlib.pyplot as plt
#matplotlib inline

from PyAedatTools.ImportAedat import ImportAedat
from PyAedatTools.ImportAedatHeaders import ImportAedatHeaders
from PyAedatTools.ImportAedatDataVersion1or2 import ImportAedatDataVersion1or2


def dvs128_events_to_frames(aedat, events_per_frame=1000, numb_frames = -1, hop_ratio=2, time_steps = 10):
    #pdb.set_trace()
    #num_frames = aedat['data']['frame']['numEvents']
    if events_per_frame:
        eventsPerFrame = events_per_frame
        num_frames = aedat['info']['numEventsInFile'] // eventsPerFrame
        num_events = aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp']
        assert(num_frames > 0)

    theEvents = []
    if numb_frames>0:
        num_frames = numb_frames
    framex = aedat['data']['polarity']['x']
    framey = aedat['data']['polarity']['y']
    framepol = aedat['data']['polarity']['polarity']
    hop_len = eventsPerFrame // hop_ratio
    frame_pos = np.arange(eventsPerFrame)
    
    #bbox1 = [42:70, 32:60]
    #bbox2 = [42:70, 70:98]
    #bbox3 = [80:108, 32:60]
    #bbox4 = [80:108, 70:98]
    bbox1 = np.empty((0,28,28))
    bbox2 = np.empty((0,28,28))
    bbox3 = np.empty((0,28,28))
    bbox4 = np.empty((0,28,28))
    
    for frame_counter in range(0, num_frames-1):
        theEvents = []
        frame = []
        theEvents = np.array([framex[frame_pos], framey[frame_pos], framepol[frame_pos]])
        frame = np.zeros([128,128])
        for pixel in range(0,theEvents.shape[1]):
            if theEvents[2,pixel]:
                frame[theEvents[0,pixel],theEvents[1,pixel]] = 1
            else:
                frame[theEvents[0,pixel],theEvents[1,pixel]] = -1
        # Extract individual pools from the frame
        #pdb.set_trace()
        
        bbox1 = np.append(bbox1,[frame[42:70, 32:60]], axis=0)
        bbox2 = np.append(bbox2,[frame[42:70, 70:98]], axis=0)
        bbox3 = np.append(bbox3,[frame[80:108, 32:60]], axis=0)
        bbox4 = np.append(bbox4,[frame[80:108, 70:98]], axis=0)
        #plot_bboxes(bbox1[-1][:][:],bbox2[-1][:][:],bbox3[-1][:][:],bbox4[-1][:][:])
        frame_pos = frame_pos+hop_len

    #pdb.set_trace()
    trunc_len = bbox1.shape[0] - bbox1.shape[0]%time_steps
    new_dim = [bbox1.shape[0] // time_steps, time_steps, 28, 28]
    bbox1 = bbox1[0:trunc_len,:,:].reshape(new_dim)
    bbox2 = bbox2[0:trunc_len,:,:].reshape(new_dim)
    bbox3 = bbox3[0:trunc_len,:,:].reshape(new_dim)
    bbox4 = bbox4[0:trunc_len,:,:].reshape(new_dim)
    
    y1 = np.full(bbox1.shape[0],1)
    y2 = np.full(bbox1.shape[0],2)
    y3 = np.full(bbox1.shape[0],3)
    y4 = np.full(bbox1.shape[0],4)
    X = np.concatenate((bbox1,bbox2,bbox3,bbox4),axis=0)
    Y = np.concatenate((y1,y2,y3,y4),axis=0)
    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    X_train = X[0:X.shape[0]//2]
    X_test = X[X.shape[0]//2:]
    Y_train = Y[0:Y.shape[0]//2]
    Y_test = Y[Y.shape[0]//2:]
    #pdb.set_trace()
    return (X_train, Y_train), (X_test, Y_test)

def plot_bboxes(bbox1,bbox2,bbox3,bbox4):
    plt.figure()
    plt.subplot(221)
    plt.imshow(bbox1)
    plt.subplot(222)
    plt.imshow(bbox2)
    plt.subplot(223)
    plt.imshow(bbox3)
    plt.subplot(224)
    plt.imshow(bbox4)
    plt.show()

def get_aed_frames(input_file_path, events_per_frame=1000, num_frames=-1, hop_ratio = 2, time_steps = 10):
    # Configure the reading parameters
    #input_file_path = '/Users/twelsh/Neuromorph2018/meltpool_data/multiwell_fixture/DVS128-2016-09-24T22-13-51-0600-0293-0_12_Hz_gallium.aedat'
    aedat = {}
    aedat['importParams'] = {}
    aedat['importParams']['filePath'] = input_file_path
    # Invoke the function
    aedat = ImportAedat(aedat)
    return dvs128_events_to_frames(aedat, events_per_frame, num_frames, hop_ratio, time_steps)
    

