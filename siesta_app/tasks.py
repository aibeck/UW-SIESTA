# Standard Library Imports
import datetime
from math import floor, ceil
import io
import os
import uuid
import joblib

# Third-Party Imports
import numpy as np
import pandas as pd
import scipy
import scipy.signal
from mne.io import read_raw_edf
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import math

# Django Imports
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse
from django.core.files.base import ContentFile

# Custom Imports
from celery import signals
from SIESTA.celery import app as celery_app
from siesta_app.utilities import model_training,  ML_predict

MEDIA = f'{settings.MEDIA_ROOT}'
STATIC = f'{settings.STATIC_ROOT}'

from SIESTA.celery import app as celery_app

__all__ = ('celery_app',)

from celery import signals

#Feature extraction functions
# Calculates power spectral density
@celery_app.task
def calculate_psd_and_f(signal,fs,epoch):
    epoch = epoch*fs
    corr_signal = signal[:len(signal)-(len(signal)%epoch)]
    new_signal = np.reshape(corr_signal,(len(corr_signal)//epoch,epoch))
    fr,p = scipy.signal.welch(new_signal,fs=fs,nperseg=fs*10,scaling='spectrum')
    return fr,p

# Performs butter bandpass filter
@celery_app.task
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# no, I don't know why this function couldn't be combined to the one above
@celery_app.task
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Creates matrix of extracted features
@celery_app.task
def feature_calculations(signal, signal_label,epoch,fs,is_ecog=True):

    signal = [max(min(x, 400), -400) for x in signal]

    fr,p = calculate_psd_and_f(signal,fs,epoch)
    epoch = epoch*fs #

    #Data from ECoG

    if is_ecog:
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

    if is_ecog:
        feature_list = [delta,deltaenergy,deltaamp, thetacon, thetaenergy, thetaamp, theta1, theta1energy,
                        theta1amp, theta2, theta2energy, theta2amp, theta3, theta3energy, theta3amp, beta,
                        betaenergy, betaamp, alpha, alfaenergy, alfaamp, sigma, sigmaenergy, sigmaamp,
                        spindle, spindleenergy, spindleamp, gamma, gammaenergy, gammaamp, EEGrel1, EEGrel2,
                        spindelhan, spectral_edge, spectral_mean50, zerocross, maxs, peaktopeak, arv,
                        rms, amplitude, amplitude_m, signal_var, skew, kurt, spectral_mean, spectral_entropy]
    else:
        feature_list = [amplitude,signal_var,skew,kurt,rms, spectral_mean,spectral_entropy,amplitude_m]

    return feature_list

@celery_app.task
# Opens .edf file containing signal data and extracts ecog and emg data
def CreateFeaturesDataFrame(self, filepath, ECoG_chan, EMG_chan, fs):
    epoch = 10

    # opens instance of .edf file; saves ecog and emg data using user-inputted channel number
    f = read_raw_edf(os.path.join(MEDIA, filepath))
    amp_value = 10 ** (1 - math.floor(math.log10(np.mean(np.abs(f.get_data(0).flatten())))))

    # gets time information from .edf file
    start_date = f.info['meas_date']
    fd = len(f)/fs
    file_end = start_date + datetime.timedelta(seconds = fd)

    num_epochs = floor(fd/epoch)
    num_epochs_each_iteration = 500

    #"Total epochs =", floor(num_epochs/num_epochs_each_iteration))

    curr_features = []
    feature_matrix = []

    for curr_epoch in range(0, ceil(num_epochs/num_epochs_each_iteration), 1):
        offset = curr_epoch*num_epochs_each_iteration

        if (num_epochs - offset) < num_epochs_each_iteration:
            num_epochs_each_iteration = num_epochs - offset

        offset_epoch = offset*fs*epoch

        ecog_features = np.column_stack(feature_calculations(f.get_data(ECoG_chan, start = offset_epoch, stop = offset_epoch + fs*epoch*num_epochs_each_iteration).flatten()*amp_value,'ECoG',epoch,fs))
        emg_features = np.column_stack(feature_calculations(f.get_data(EMG_chan, start = offset_epoch, stop = offset_epoch + fs*epoch*num_epochs_each_iteration).flatten()*amp_value,'EMG',epoch,fs,False))

        curr_features = np.c_[list(range(0+offset, num_epochs_each_iteration+offset)), ecog_features, emg_features]
        feature_matrix.extend(curr_features)

        curr_state = "{}".format(floor(100*offset/num_epochs))
        self.update_state(state=curr_state, meta = {'status':'progressing'})

    del(filepath)

    # creates time step (in seconds) according to user-inputted epoch length
    step = datetime.timedelta(seconds=epoch)

    # iterates by time step until end of file
    time_stamps = []
    while start_date < file_end:
        time_stamps.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        start_date += step

    feature_labels = ['Epoch','ECoG_delta','ECoG_deltaenergy','ECoG_deltaamp','ECoG_thetacon','ECoG_thetaenergy','ECoG_thetaamp', 'ECoG_theta1','ECoG_theta1energy',
                      'ECoG_theta1amp','ECoG_theta2', 'ECoG_theta2energy','ECoG_theta2amp', 'ECoG_theta3', 'ECoG_theta3energy','ECoG_theta3amp', 'ECoG_beta',
                      'ECoG_betaenergy','ECoG_betaamp','ECoG_alpha', 'ECoG_alfaenergy', 'ECoG_alfaamp', 'ECoG_sigma', 'ECoG_sigmaenergy', 'ECoG_sigmaamp',
                      'ECoG_spindle', 'ECoG_spindlenergy', 'ECoG_spindleamp', 'ECoG_gamma', 'ECoG_gammaenergy', 'ECoG_gammaamp', 'ECoG_rel1', 'ECoG_rel2',
                      'ECoG_spindelhan', 'ECoG_spectral_edge', 'ECoG_spectral_mean50', 'ECoG_zerocross', 'ECoG_maxs' , 'ECoG_peaktopeak', 'ECoG_arv',
                      'ECoG_rms', 'ECoG_amplitude', 'ECoG_amplitude_m', 'ECoG_signal_var', 'ECoG_skew', 'ECoG_kurt', 'ECoG_spectral_mean', 'ECoG_spectral_entropy',
                      'EMG_amplitude','EMG_signal_var','EMG_skew','EMG_kurt','EMG_rms','EMG_spectral_mean','EMG_spectral_entropy','EMG_amplitude_m']
    data = pd.DataFrame(feature_matrix,columns=feature_labels)
    final_data = pd.DataFrame(data.iloc[:,1:])

    if (len(final_data) != len(time_stamps)):
        time_stamps = time_stamps[:len(final_data)]
    final_data['time_stamps'] = time_stamps

    return final_data


@celery_app.task(bind=True)
def download_feat(self, filepath, fs, ECoG_chan, EMG_chan):
    data_frame = CreateFeaturesDataFrame(self,filepath,ECoG_chan,EMG_chan,fs)

    return data_frame.to_json()


@celery_app.task(bind=True)
def train_new_model(self, filepaths, include_training_data):
    data = pd.DataFrame()

    if include_training_data:
        data = pd.concat([data, pd.read_csv(os.path.join(STATIC, "admin/model_data/training_dataset.csv"))])

    if (len(filepaths) > 0):
        for f in filepaths:
            data = pd.concat([data, pd.read_csv(os.path.join(MEDIA, f))])

    if (len(data) == 0):
        print("throw some error 2")

    self.update_state(state='FITTING', meta = {'status':'fitting'})

    data.dropna(inplace=True)
    new_model = model_training(data)

    if (len(filepaths) > 0):
        for f in filepaths:
            del(f)

    modelName = str(uuid.uuid4()) + '.file'
    joblib.dump(new_model, os.path.join(MEDIA, modelName), compress=3)
    return modelName


@celery_app.task(bind=True)
def sleep_prediction(self, csv_cache_key, model_cache):
    try:
        data = pd.read_csv(io.BytesIO(caches['cloud'].get(csv_cache_key)), parse_dates=[-1])
        if data is None:
            raise ValueError("CSV data not found in cache.")

        scaling_cache = caches['cloud'].get('scaling')

        if model_cache is None or scaling_cache is None:
            raise ValueError("Model or scaling data not found.")

        with open(model_cache, 'rb') as f:
            models = joblib.load(f)
        with open(scaling_cache, 'rb') as f:
            scaling = joblib.load(f)

        scored_data = ML_predict(data, models, scaling)
        if scored_data is None:
            raise ValueError("Error in prediction.")

        caches['cloud'].delete(csv_cache_key)
        return scored_data.to_json(orient='records')
    except Exception as e:
        # Log error and re-raise exception
        raise


@signals.worker_ready.connect
def download_ML(**kwargs):
    model_cache_key = 'trained_model'
    model = os.path.join(STATIC, "admin/model_data/trained_model.file")
    scaling = os.path.join(STATIC, "admin/model_data/scaling.file")

    caches['cloud'].set(model_cache_key, model, None)
    caches['cloud'].set('scaling', scaling, None)