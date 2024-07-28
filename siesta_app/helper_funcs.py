from sklearn import preprocessing

import scipy
import scipy.signal
from scipy.signal import butter, lfilter

import pandas as pd
import numpy as np

import pyedflib
import datetime

from math import floor

    
#Feature extraction functions
# Calculates power spectral density
def calculate_psd_and_f(signal,fs,epoch):
    epoch = epoch*fs
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    fr,p = scipy.signal.welch(new_signal,fs=fs,nperseg=fs*10,scaling='spectrum')
    return fr,p

# Performs butter bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# no, I don't know why this function couldn't be combined to the one above
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Creates matrix of extracted features
def feature_calculations(signal,signal_label,epoch,fs,is_eeg=True):

    fr,p = calculate_psd_and_f(signal,fs,epoch)
    epoch = epoch*fs #

    #Data from EEG

    if is_eeg:
        max_fr = 50

        ## Calculate the total power, total power per epoch, and extract the relevant frequencies.
        ## IMPORTANT NOTE: These are not the ACTUAL power values, they are standardized to account
        ## for individual variability, and are thus relative.
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T

        ## Calculate the relative power at the different brain waves:
        delta = np.sum(std_power[:,(freq>=0.5) & (freq <=4)],axis=1)


        thetacon = np.sum(std_power[:,(freq>=4) & (freq <=12)],axis=1)
        theta1 = np.sum(std_power[:,(freq>=6) & (freq <=9)],axis=1)
        theta2 = np.sum(std_power[:,(freq>=5.5) & (freq <=8.5)],axis=1)
        theta3 = np.sum(std_power[:,(freq>=7) & (freq <=10)],axis=1)

        beta = np.sum(std_power[:,(freq>=20) & (freq <=40)],axis=1)

        alpha = np.sum(std_power[:,(freq>=8) & (freq <=13)],axis=1)
        sigma = np.sum(std_power[:,(freq>=11) & (freq <=15)],axis=1)
        spindle = np.sum(std_power[:,(freq>=12) & (freq <=14)],axis=1)
        gamma= np.sum(std_power[:,(freq>=35) & (freq <=45)],axis=1)

        temp1= np.sum(std_power[:,(freq>=0.5) & (freq <=20)],axis=1)
        temp2= np.sum(std_power[:,(freq>=0.5) & (freq <=50)],axis=1)

        temp3= np.sum(std_power[:,(freq>=0.5) & (freq <=40)],axis=1)
        temp4= np.sum(std_power[:,(freq>=11) & (freq <=16)],axis=1)


        EEGrel1 = thetacon/delta;
        EEGrel2 = temp1/temp2;
        EEGrel3 = temp4/temp3;

        hann = np.hanning(12);

        spindelhan1=np.convolve(hann,EEGrel3,'same');

        spindelhan=np.transpose(spindelhan1);

        ## Calculate the 90% spectral edge:
        spectral90 = 0.9*(np.sum(sum_power,axis=1))
        s_edge = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_edge[row_ind,:]) if j>=spectral90[row_ind]][0] for row_ind in range(s_edge.shape[0])]
        spectral_edge = np.take(fr,l) # spectral edge 90%, the frequency below which power sums to 90% of the total power

         ## Calculate the 50% spectral mean:
        spectral50 = 0.5*(np.sum(sum_power,axis=1))
        s_mean = np.cumsum(sum_power,axis=1)
        l = [[n for n,j in enumerate(s_mean[row_ind,:]) if j>=spectral50[row_ind]][0] for row_ind in range(s_mean.shape[0])]
        spectral_mean50 = np.take(fr,l)

    else:
        #for EMG
        max_fr = 100

        ## Calculate the total power, total power per epoch, and extract the relevant frequencies:
        freq = fr[(fr>=0.5) & (fr <=max_fr)]
        sum_power = p[:,(fr>=0.5) & (fr <=max_fr)]
        max_power = np.max(sum_power,axis=1)
        min_power = np.min(sum_power,axis=1)
        range_power = max_power - min_power
        std_power = ((sum_power.T-min_power)/range_power).T

    ## Calculate the Root Mean Square of the signal
    signal = signal[0:p.shape[0]*epoch]
    s = np.reshape(signal,(p.shape[0],epoch))
    rms = np.sqrt(np.mean((s)**2,axis=1)) #root mean square

    ## Calculate amplitude and spectral variation:
    amplitude = np.mean(np.abs(s),axis=1)
    amplitude_m=np.median(np.abs(s),axis=1)
    signal_var = (np.sum((np.abs(s).T - np.mean(np.abs(s),axis=1)).T**2,axis=1)/(len(s[0,:])-1)) # The variation

    ## Calculate skewness and kurtosis
    m3 = np.mean((s-np.mean(s))**3,axis=1) #3rd moment
    m2 = np.mean((s-np.mean(s))**2,axis=1) #2nd moment
    m4 = np.mean((s-np.mean(s))**4,axis=1) #4th moment
    skew = m3/(m2**(3/2)) # skewness of the signal, which is a measure of symmetry
    kurt = m4/(m2**2) #kurtosis of the signal, which is a measure of tail magnitude

    ## Calculate more time features
    signalzero=preprocessing.maxabs_scale(s,axis=1)
    zerocross = (np.diff(np.sign(signalzero)) != 0).sum(axis=1)

    maxs = np.amax(s,axis=1)
    mins = np.amin(s,axis=1)

    peaktopeak= maxs - mins

    arv = np.mean(np.abs(s),axis=1)

    #Energy and amplitud    e
    deltacomp = butter_bandpass_filter(s, 0.5, 4, fs, 5)
    deltaenergy = sum([x*2 for x in np.matrix.transpose(deltacomp)])
    deltaamp = np.mean(np.abs(deltacomp),axis=1)

    thetacomp = butter_bandpass_filter(s, 4, 12, fs, 5)
    thetaenergy = sum([x*2 for x in np.matrix.transpose(thetacomp)])
    thetaamp = np.mean(np.abs(thetacomp),axis=1)


    theta1comp = butter_bandpass_filter(s, 6, 9, fs, 5)
    theta1energy = sum([x*2 for x in np.matrix.transpose(theta1comp)])
    theta1amp = np.mean(np.abs(theta1comp),axis=1)

    theta2comp = butter_bandpass_filter(s, 5.5, 8.5, fs, 5)
    theta2energy = sum([x*2 for x in np.matrix.transpose(theta2comp)])
    theta2amp = np.mean(np.abs(theta2comp),axis=1)

    theta3comp = butter_bandpass_filter(s, 7, 10, fs, 5)
    theta3energy = sum([x*2 for x in np.matrix.transpose(theta3comp)])
    theta3amp = np.mean(np.abs(theta3comp),axis=1)

    betacomp = butter_bandpass_filter(s, 20, 40, fs, 5)
    betaenergy = sum([x*2 for x in np.matrix.transpose(betacomp)])
    betaamp = np.mean(np.abs(betacomp),axis=1)

    alfacomp = butter_bandpass_filter(s, 8, 13, fs, 5)
    alfaenergy = sum([x*2 for x in np.matrix.transpose(alfacomp)])
    alfaamp = np.mean(np.abs(alfacomp),axis=1)

    sigmacomp = butter_bandpass_filter(s, 11, 15, fs, 5)
    sigmaenergy = sum([x*2 for x in np.matrix.transpose(sigmacomp)])
    sigmaamp = np.mean(np.abs(sigmacomp),axis=1)

    spindlecomp = butter_bandpass_filter(s, 12, 14, fs, 5)
    spindleenergy = sum([x*2 for x in np.matrix.transpose(spindlecomp)])
    spindleamp = np.mean(np.abs(spindlecomp),axis=1)

    gammacomp = butter_bandpass_filter(s, 35, 45, fs, 5)
    gammaenergy = sum([x*2 for x in np.matrix.transpose(gammacomp)])
    gammaamp = np.mean(np.abs(gammacomp),axis=1)

    ## Calculate the spectral mean and the spectral entropy (essentially the spectral power distribution):
    spectral_mean = np.mean(std_power,axis=1)
    spectral_entropy = -(np.sum((std_power+0.01)*np.log(std_power+0.01),axis=1))/(np.log(len(std_power[0,:])))

    if is_eeg:
        feature_list = [delta,deltaenergy,deltaamp, thetacon, thetaenergy, thetaamp, theta1, theta1energy,
                        theta1amp, theta2, theta2energy, theta2amp, theta3, theta3energy, theta3amp, beta,
                        betaenergy, betaamp, alpha, alfaenergy, alfaamp, sigma, sigmaenergy, sigmaamp,
                        spindle, spindleenergy, spindleamp, gamma, gammaenergy, gammaamp, EEGrel1, EEGrel2,
                        spindelhan, spectral_edge, spectral_mean50, zerocross, maxs, peaktopeak, arv,
                        rms, amplitude, amplitude_m, signal_var, skew, kurt, spectral_mean, spectral_entropy]
    else:
        feature_list = [amplitude,signal_var,skew,kurt,rms, spectral_mean,spectral_entropy,amplitude_m]
        
    return feature_list


# Opens .edf file containing signal data and extracts eeg and emg data
def CreateFeaturesDataFrame(self, file_name,EEG_chan,EMG_chan,epoch,fs):

    self.update_state(state='PENDING')

    # opens instance of .edf file; saves eeg and emg data using user-inputted channel number
    f = pyedflib.EdfReader(file_name)
    print(f)
    print("edf file")
    
    # gets time information from .edf file
    start_date = datetime.datetime(f.getStartdatetime().year,f.getStartdatetime().month,f.getStartdatetime().day,
                                   f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second)
    fd = f.getFileDuration()
    print(fd)
    file_end = start_date + datetime.timedelta(seconds = fd)
    print(file_end)
    
    num_epochs = floor(fd/epoch)
    num_epochs_each_iteration = 50
    
    print("Total epochs =", floor(num_epochs/num_epochs_each_iteration))
    
    curr_features = []
    feature_matrix = []
    
    for curr_epoch in range(0, floor(num_epochs/num_epochs_each_iteration), 1):
        offset = curr_epoch*num_epochs_each_iteration
        offset_epoch = offset*fs*epoch
    
        eeg_features = np.column_stack(feature_calculations(f.readSignal(EEG_chan, offset_epoch, fs*epoch*num_epochs_each_iteration),'EEG',epoch,fs))
        emg_features = np.column_stack(feature_calculations(f.readSignal(EMG_chan, offset_epoch, fs*epoch*num_epochs_each_iteration),'EMG',epoch,fs,False))
        
        curr_features = np.c_[list(range(0+offset, 50+offset)), eeg_features, emg_features]
        feature_matrix.extend(curr_features)
        print(len(feature_matrix))
    
    f._close()

    # creates time step (in seconds) according to user-inputted epoch length
    step = datetime.timedelta(seconds=epoch)

    # iterates by time step until end of file
    time_stamps = []
    while start_date < file_end:
        time_stamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        start_date += step
        
    feature_labels = ['Epoch','delta','deltaenergy','deltaamp','thetacon','thetaenergy','thetaamp', 'theta1','theta1energy',
                      'theta1amp','theta2', 'theta2energy','theta2amp', 'theta3', 'theta3energy','theta3amp', 'beta',
                      'betaenergy','betaamp','alpha', 'alfaenergy', 'alfaamp', 'sigma', 'sigmaenergy', 'sigmaamp',
                      'spindle', 'spindlenergy', 'spindleamp', 'gamma', 'gammaenergy', 'gammaamp', 'EEGrel1', 'EEGrel2',
                      'spindelhan', 'spectral_edge', 'spectral_mean50', 'zerocross', 'maxs' , 'peaktopeak', 'arv',
                      'rms', 'amplitude', 'amplitude_m', 'signal_var', 'skew', 'kurt', 'spectral_mean', 'spectral_entropy',
                      'EMG_amplitude','EMG_signal_var','EMG_skew','EMG_kurt','EMG_rms','EMG_spectral_mean','EMG_spectral_entropy','EMG_amplitude_m']
    data = pd.DataFrame(feature_matrix,columns=feature_labels)
    final_data = pd.DataFrame(data.iloc[:,1:])
    
    if len(final_data) != len(time_stamps):
        time_stamps = time_stamps[:len(final_data)]
    final_data['time_stamps'] = time_stamps

    self.update_state(state='COMPLETE')

    return final_data