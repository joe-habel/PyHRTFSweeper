from scipy.io import loadmat
from scipy.spatial import Delaunay
from scipy.signal import butter, lfilter, convolve
import scipy.io.wavfile as wav
from pysndfx import AudioEffectsChain
from numba import jit
import numpy as np
import time
import ffmpeg
import os
import shutil


import warnings

warnings.filterwarnings("ignore")

start= time.time()

class HRTF(object):
    def __init__(self):
        self.hrir = {}
        self.triangulation = {'points': [],
                              'triangles' : None}
    def weight_calc(self,points):
        tri = self.triangulation['triangles'].find_simplex(points)
        X = self.triangulation['triangles'].transform[tri,:2]
        Y = points - self.triangulation['triangles'].transform[tri,2]
        b = np.einsum('ijk,ik->ij', X, Y)
        return (np.c_[b,1-b.sum(axis=1)],
                self.triangulation['triangles'].simplices[tri])
    
    
    
    def load_subject(self,subject_file,hrir_len=200,azimuths=None,elevations=None):
        x = loadmat(subject_file)
        hrir_r = x['hrir_r']
        hrir_l = x['hrir_l'] 
        ir = {'L': {}, 'R': {}}
        if azimuths is None:
            azimuths = [-80, -65, -55, -45, -40, -35, -30, -25, -20,
					-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80]
        if elevations is None:
            elevations = [-45+5.625*x for x in range(50)]
        points = []

        for azi in [-90,90]:
            ir['L'][azi] = {}
            ir['R'][azi] = {}
            if azi == -90:
                for j,elv in enumerate(elevations):
                    if j == 0:
                        to_add_l = hrir_l[0,j]
                        to_add_r = hrir_r[0,j]
                    else:
                        to_add_l += hrir_l[0,j]
                        to_add_r += hrir_r[0,j]
                avg_l = to_add_l/50
                avg_r = to_add_r/50
                all_elvs = [x for x in elevations]
                all_elvs.extend([-90,270])
                for elv in all_elvs:
                    ir['L'][azi][elv] = avg_l
                    ir['R'][azi][elv] = avg_r
                    points.append([azi,elv])
            else:
                for j,elv in enumerate(elevations):
                    if j == 0:
                        to_add_l = hrir_l[-1,j]
                        to_add_r = hrir_r[-1,j]
                    else:
                        to_add_l += hrir_l[-1,j]
                        to_add_r += hrir_r[-1,j]
                avg_l = to_add_l/50
                avg_r = to_add_r/50
                for elv in all_elvs:
                    ir['L'][azi][elv] = avg_l
                    ir['R'][azi][elv] = avg_r
                    points.append([azi,elv])

        for i,azi in enumerate(azimuths):
            ir['L'][azi] = {}
            ir['R'][azi] = {}
            
            ir['L'][azi][-90] = calculate_weighted(hrir_l[i,0],-45.0,
                                                   hrir_l[i,-1],230.625)
            ir['R'][azi][-90] = calculate_weighted(hrir_r[i,0],-45.0,
                                                   hrir_r[i,-1],230.625)
            points.append([azi,-90])
            for j,elv in enumerate(elevations):
                ir['L'][azi][elv] = hrir_l[i,j]
                ir['R'][azi][elv] = hrir_r[i,j]
                points.append([azi,elv])
            ir['L'][azi][270] = ir['L'][azi][-90]
            ir['R'][azi][270] = ir['R'][azi][-90]
            points.append([azi,270])
        
        self.hrir = ir
        self.triangulation['triangles'] = Delaunay(np.array(points))
        self.triangulation['points'] = points


def check_file_type(audio_file):
    _, ext = os.path.splitext(audio_file)
    return ext

def other_to_wav(audio_file):
    base = os.path.basename(audio_file)
    filename , _ = os.path.splitext(base)
    if os.path.exists(filename + '.wav'):
        return filename
    (
     ffmpeg
     .input(audio_file)
     .output(filename + '.wav')
     .run()
    )
    return filename


def calculate_weighted(vect1,point1,vect2,point2):
    total = abs(-90 - point1) + abs(270 - point2)
    weight1 = abs(-90 - point1)/total
    weight2 = abs(270 - point2)/total
    vect1 *= weight1
    vect2 *= weight2
    return vect1+vect2
         

def interpolater(angles,hrtf):
    weights,indices = hrtf.weight_calc(angles)
    points = []
    for triangle in indices:
        triangle_points = []
        for index in triangle:
            triangle_points.append(hrtf.triangulation['points'][index])
        points.append(triangle_points)
    right = hrtf.hrir['R']
    left = hrtf.hrir['L']
    interp_R = []
    interp_L = []
    @jit
    def interpolate_loop(right,left,weights,points,interp_R,interp_L):
        for i in range(weights.shape[0]):
            interp_R.append(weights[i][0]*right[points[i][0][0]][points[i][0][1]]+
                            weights[i][1]*right[points[i][1][0]][points[i][1][1]]+
                            weights[i][2]*right[points[i][2][0]][points[i][2][1]])
    
            interp_L.append(weights[i][0]*left[points[i][0][0]][points[i][0][1]]+
                            weights[i][1]*left[points[i][1][0]][points[i][1][1]]+
                            weights[i][2]*left[points[i][2][0]][points[i][2][1]])
    interpolate_loop(right,left,weights,points,interp_R,interp_L)
    return (interp_R,interp_L)

def to_mono(audio):
    audio = audio.astype(np.float32)
    one_channel = audio.sum(axis=1)/4
    return one_channel.astype(np.int16)

def butter_pass(cutoff,sr,filt_type,order=5):
    nyq = 0.5*sr
    cutoff = cutoff / nyq
    b, a = butter(order,cutoff, btype=filt_type, analog=False)
    return b,a

def butter_lowpass_filter(data,lowcutoff,sr,order=5):
    b,a = butter_pass(lowcutoff,sr,'low',order=order)
    y = lfilter(b,a,data)
    return y

def butter_highpass_filter(data,highcutoff,sr,order=5):
    b,a = butter_pass(highcutoff,sr,'high',order)
    y = lfilter(b,a,data)
    return y

def add_reverb(in_file,out_file,room_scale=50):
    shutil.move(in_file,'in.wav')
    fx = (AudioEffectsChain().reverb(room_scale=room_scale))
    fx('in.wav','out.wav')
    shutil.move('out.wav',out_file)
    shutil.move('in.wav',in_file)
    
    
def crossfade_tails(left,right,tailed_left,tailed_right,filter_len):
    final_left = []
    final_right = []
    t = np.linspace(0,np.pi/2,filter_len)
    fade_out = np.cos(t)**2
    fade_in = np.sin(t)**2
    for i, left_block in enumerate(left):
        if i == 0:
            final_left.extend(left_block)
            final_right.extend(right[i])
        else:
            if len(left_block) < filter_len:
                t = np.linspace(0,np.pi/2,len(left_block))
                filter_len = len(left_block)
                fade_out = np.cos(t)**2
                fade_in = np.sin(t)**2
            faded_left = tailed_left[i-1][-filter_len:]*fade_out + left_block[:filter_len]*fade_in
            faded_right = tailed_right[i-1][-filter_len:]*fade_out + right[i][:filter_len]*fade_in
            left_block[:filter_len] = faded_left
            right[i][:filter_len] = faded_right
            final_left.extend(left_block)
            final_right.extend(right[i])
    return final_left, final_right            
            
            

def multiple_convolve(mono,hrir_l,hrir_r,audio_rate,circle_period,
                      crossfade_ms=25,low_freq = 50, high_freq = None):
    crossfade_amount = int(audio_rate*crossfade_ms/1000.)
    left_channel = []
    right_channel = []
    tailed_left = []
    tailed_right = []
    total_angles = len(hrir_l)
    i = 0
    pos = 0
    angle_time = int(circle_period/total_angles*audio_rate)
    total_samples = len(mono)
    left = butter_highpass_filter(mono,low_freq,audio_rate)
    right = butter_highpass_filter(mono,low_freq,audio_rate)
    save = butter_lowpass_filter(mono,low_freq,audio_rate)
    while pos < total_samples:
        end = pos + angle_time+199
        left_channel.append(convolve(left[pos:end],hrir_l[i],'valid'))
        right_channel.append(convolve(right[pos:end],hrir_r[i],'valid'))
        tailed_left.append(convolve(left[pos:end+crossfade_amount],
                                    hrir_l[i],'valid'))
        tailed_right.append(convolve(right[pos:end+crossfade_amount],
                                     hrir_r[i],'valid'))
        pos += angle_time
        i += 1
        i %= total_angles
    
    left_channel,right_channel = crossfade_tails(left_channel,
                                                 right_channel,tailed_left,
                                                 tailed_right,crossfade_amount)
    left_channel = np.array(left_channel)
    right_channel = np.array(right_channel)
    left_channel += save[:-199]
    right_channel += save[:-199]
    return np.column_stack((left_channel,right_channel))

def test_subject(subject_number):
    if subject_number < 10:
        subject_number = '00' + str(subject_number)
    elif subject_number < 100:
        subject_number = '0' + str(subject_number)
    else:
        subject_number = str(subject_number)
    
    subject = 'subject_' + subject_number
    return os.path.join('CIPIC_hrtf_database', 'standard_hrir_database',
                        subject,'hrir_final.mat')


def panner(audio_file,new_file_name,audio_dir,sweep_time=15,sweep_frequency=0.1,
         crossfade_ms=25,room_scale=85,subject=65,max_angle=90, low_freq=75,
         high_freq = None):
    
    if os.path.exists(os.path.join(audio_dir,'8D',new_file_name + '.wav')):
        return 
    if os.path.exists('in.wav'):
        os.remove('in.wav')
    if os.path.exists('out.wav'):
        os.remove('in.wav')
    
    test = test_subject(subject)
    x = loadmat(test)
    hrir_r = x['hrir_r']
    hrir_l = x['hrir_l']
    hrtf = HRTF()
    hrtf.load_subject(test)
    
    number_per_side = int(sweep_time/(2*sweep_frequency))
    
    if sweep_time/(number_per_side*2) < crossfade_ms/1000.:
        print ('too fast to crossfade!!!!')
    
    ext = check_file_type(audio_file)
    cvt = False
    if ext != '.wav':
        print ('...Converting Audio')
        fn = other_to_wav(audio_file)
        audio_file = fn + '.wav'
        cvt = True
        
    audio_rate,test_audio = wav.read(audio_file)
    
    mono = to_mono(test_audio)
    
    circular = np.linspace(-max_angle,max_angle,number_per_side)
    angles = [[az,0] for az in circular]
    circle_reverse = np.linspace(max_angle,-max_angle,number_per_side)
    angles.extend([[az,180] for az in circle_reverse])
    
    hrir_l, hrir_r =interpolater(angles,hrtf)
    
    print ('...Convolving Audio')
    try:
        new_audio = multiple_convolve(mono,hrir_l,hrir_r,audio_rate,sweep_time,
                                  crossfade_ms,low_freq)
    except:
        if cvt:
            os.remove(audio_file)
        print ('Bad Convolve')
        return
    largest_value =  np.max(new_audio)
    if largest_value >= 32767:
        new_audio *= 32766/largest_value
    wav.write('temp.wav',audio_rate,new_audio.astype(np.int16))

    print ('...Adding Reverb')
    add_reverb('temp.wav',os.path.join(audio_dir,'8D',new_file_name + '.wav'),room_scale)
    os.remove('temp.wav')
    if cvt:
        os.remove(audio_file)

#panner("test.mp3", 'slower','', sweep_time = 15, sweep_frequency = .1, 
#     crossfade_ms = 25, room_scale = 85, low_freq = 75,subject=163)

#print (time.time() - start)