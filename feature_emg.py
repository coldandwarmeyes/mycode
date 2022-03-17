# prepare training data for CNN models
# step 1: data downsampling, from 2048 hz to 1024 hz
# step 2: data segmentation: signals were split to segmentations of 200 ms with 50% overlap
# step 3: save training data, orgnized as subj/session/sample, each sample was reshaped as time_points *16 * 16
import numpy as np
import os
import pathlib
from settings import Config
from scipy import signal
import hyser_functions as hyser
from sklearn.preprocessing import scale
from multiprocessing import Process
def get_feature(emg_data, windou_len, step_len, fs):
    thresh=0.0004
    channel_select = np.array([0,   1,   2,   3,   4,   5,   6,   8,   9,  11,  12,  13,  14,
        15,  16,  20,  24,  25,  28,  29,  30,  31,  36,  40,  41,  42,
        47,  54,  55,  56,  57,  58,  59,  60,  61,  70,  71,  72,  73,
        74,  77,  78,  85,  86,  88,  92,  94, 100, 102, 108, 110, 111,
       115, 116, 117, 122, 123, 124, 126, 127, 131, 132, 133, 137, 142,
       146, 147, 148, 149, 153, 154, 155, 156, 157, 158, 159, 161, 162,
       163, 164, 170, 171, 172, 174, 175, 176, 177, 178, 186, 187, 188,
       190, 191, 192, 193, 194, 201, 204, 217, 218, 219, 220, 223, 234,
       239, 250, 251, 253, 254, 255])
    emg_data_channelselect = np.zeros([channel_select.shape[0], emg_data.shape[1]])

    for i in range(channel_select.shape[0]):
        emg_data_channelselect[i] = emg_data[int(channel_select[i])]

    emg_data_channelselect=emg_data_channelselect.transpose(1,0)

    mlf=hyser.get_mfl(emg_data_channelselect, windou_len, step_len, fs)
    # wa=hyser.get_wa(emg_data_channelselect, windou_len, step_len, fs)
    # vare=hyser.get_vare(emg_data_channelselect, windou_len, step_len, fs)
    ssi=hyser.get_ssi(emg_data_channelselect, windou_len, step_len, fs)
    # myop=hyser.get_myop(emg_data_channelselect, windou_len, step_len, fs)
    # mmav2=hyser.get_mmav2(emg_data_channelselect, windou_len, step_len, fs)
    # mmav=hyser.get_mmav(emg_data_channelselect, windou_len, step_len, fs)
    # ld=hyser.get_ld(emg_data_channelselect, windou_len, step_len, fs)
    dasdv=hyser.get_dasdv(emg_data_channelselect, windou_len, step_len, fs)
    # aac=hyser.get_aac(emg_data_channelselect, windou_len, step_len, fs)
    # rms=hyser.get_rms(emg_data_channelselect, windou_len, step_len, fs)
    # wl=hyser.get_wl(emg_data_channelselect, windou_len, step_len, fs)
    zc=hyser.get_zc(emg_data_channelselect, windou_len, step_len, thresh=thresh, fs=fs)
    ssc=hyser.get_ssc(emg_data_channelselect, windou_len, step_len, thresh=thresh, fs=fs)
    # mav=hyser.get_mav(emg_data_channelselect, windou_len, step_len, fs)
    # iemg=hyser.get_iemg(emg_data_channelselect, windou_len, step_len, fs)
    ae=hyser.get_ae(emg_data_channelselect, windou_len, step_len, fs)
    # var=hyser.get_var(emg_data_channelselect, windou_len, step_len, fs)
    # sd=hyser.get_sd(emg_data_channelselect, windou_len, step_len, fs)
    # cov=hyser.get_cov(emg_data_channelselect, windou_len, step_len, fs)
    # kurt=hyser.get_kurt(emg_data_channelselect, windou_len, step_len, fs)
    # skew=hyser.get_skew(emg_data_channelselect, windou_len, step_len, fs)
    # iqr=hyser.get_iqr(emg_data_channelselect, windou_len, step_len, fs)
    # mad=hyser.get_mad(emg_data_channelselect, windou_len, step_len, fs)
    # # ar = hyser.get_ar(emg_data_channelselect, windou_len, step_len, fs)
    # damv=hyser.get_damv(emg_data_channelselect, windou_len, step_len, fs)
    # tm=hyser.get_tm(emg_data_channelselect, windou_len, step_len, fs)
    vo=hyser.get_vo(emg_data_channelselect, windou_len, step_len, fs)
    dvarv=hyser.get_dvarv(emg_data_channelselect, windou_len, step_len, fs)
    # ldamv=hyser.get_ldamv(emg_data_channelselect, windou_len, step_len, fs)
    ldasdv=hyser.get_ldasdv(emg_data_channelselect, windou_len, step_len, fs)
    # card=hyser.get_card(emg_data_channelselect, windou_len, step_len, fs)
    # lcov=hyser.get_lcov(abs(emg_data_channelselect), windou_len, step_len, fs)
    ltkeo=hyser.get_ltkeo(emg_data_channelselect, windou_len, step_len, fs)
    # msr=hyser.get_msr(abs(emg_data_channelselect), windou_len, step_len, fs)
    # ass=hyser.get_ass(abs(emg_data_channelselect), windou_len, step_len, fs)
    # asm=hyser.get_asm(abs(emg_data_channelselect), windou_len, step_len, fs)
    # fzc=hyser.get_fzc(emg_data_channelselect, windou_len, step_len, fs)
    # ewl=hyser.get_ewl(emg_data_channelselect, windou_len, step_len, fs)
    # emav=hyser.get_emav(abs(emg_data_channelselect), windou_len, step_len, fs)
    # feature=np.concatenate((mlf, wa, vare, ssi, myop, mmav2, mmav, ld, dasdv, aac, rms, wl, zc, ssc, mav, iemg, ae, var, sd, cov, kurt, skew, iqr, mad, damv, tm, vo, dvarv, ldamv, ldasdv, card, lcov, ltkeo, msr, ass, asm, fzc,ewl,emav),axis=0)
    # feature=np.concatenate((zc, ssc, dasdv, ldasdv, ltkeo),axis=0)
    feature = np.concatenate((zc, ssc, dasdv, ldasdv, ltkeo, mlf, dvarv, ssi, ae, vo), axis=0)
    feature = feature.reshape(-1)

    return feature

def gen_training_data(subj_id):
    FS=2048
    SEGMENTATION_LENGTH= 0.25
    OVERLAP_RATIO=0.5
    DISCARD_LENGTH=0.25
    FFT_SIZE = 256
    U=1024
    # 10 class EMG
    index=np.array([6, 7, 8, 9, 10, 11, 30, 31, 32, 34])
    index=index-1

    # real location
    a=np.arange(64)
    a1=a[::-1].reshape(8,8)
    a2=a1+64
    a3=a2+64
    a4=a3+64
    extensor_muscles=np.concatenate((a1,a2),axis=0)
    flexors=np.concatenate((a3,a4),axis=0)
    map=np.concatenate((flexors,extensor_muscles),axis=1)
    map=map.reshape(-1)


    for session_id in range(2):
        counter = [0] * 34
        file_list = os.listdir(os.path.join(SAVE_PATH, MODES, f"subject_{subj_id}", f"session_{session_id}"))
        # file_list.sort(key=lambda x:int(x.replace(".npz","")))
        # for idx , file_name in enumerate(file_list):
        file_name = file_list[0]
        with np.load(os.path.join(SAVE_PATH,MODES,f"subject_{subj_id}",f"session_{session_id}",file_name),allow_pickle=True) as f:
            sig=f["x"]
            sig=sig.reshape(-1)[0]
            sig_key_name=list(sig.keys())[-1]
            sig=sig[sig_key_name]
            sig=np.array(sig)
            sig = sig[0]
            sig = [np.array(c) for c in sig]
            sig = np.array(sig)
            sample_number, time_point, n_channel = sig.shape
            for j in range(sample_number):
                sub_sig=sig[j,:,:]
                sub_sig=sub_sig.transpose(1,0)
            # sig=sig.transpose(1,0) # transpose to n_channel * time_point
                sub_sig=signal.decimate(sub_sig,q=2,axis=1) # downsample, FS=2048/2
                time_point=sub_sig.shape[1]
                FS=time_point
            # n_channel,time_point=sig.shape
            # sig=scale(sig,axis=1) # Z-score normalization
                sub_sig_map=np.zeros(sub_sig.shape)



                for k in range(n_channel):
                    sub_sig_map[k]=sub_sig[int(map[k])]
                label=f["y"]
                label=label[0]



                # n_segments=int((time_point-SEGMENTATION_LENGTH*FS)/(OVERLAP_RATIO*SEGMENTATION_LENGTH*FS)+1)
                # step=int((1-OVERLAP_RATIO)*SEGMENTATION_LENGTH*FS)
                # real_start_idx=int(DISCARD_LENGTH*FS//step)
                if label[j] in index:
                    # for i in range(real_start_idx,n_segments):
                    # sub_sig=sub_sig[:,i*step:i*step+int(SEGMENTATION_LENGTH*FS)] # split segmentse
                    # nf, taxis, zxx = signal.stft(seg, nperseg=FFT_SIZE, fs=FS, noverlap=FFT_SIZE // 2, padded=False,axis=1)
                    # zxx=abs(zxx)
                    # zxx=np.sign(sub_sig_map)*np.log(1+U*abs(sub_sig_map))/np.log(1+U)
                    feature=get_feature(sub_sig_map,SEGMENTATION_LENGTH,OVERLAP_RATIO*SEGMENTATION_LENGTH,FS)
                    # zxx=zxx.reshape(-1,16,16)
                    # # discard components higher than 500 Hz
                    # zxx=zxx[:,0:64,:].reshape(16,16,-1).astype(np.float32)
                    # zxx=np.transpose(zxx,(2,0,1))  # reshape to topographic images to train
                    # seg=seg.transpose(1,0).reshape((-1,16,16))
                    current_save_path=os.path.join("D:\TL\deepforest",f"deepforest_top10_DF",f"subject_{subj_id}",f"session_{session_id}",f"{label[j]}",f"{counter[label[j]]}")
                    pathlib.Path(current_save_path).mkdir(parents=True,exist_ok=True)
                    np.savez(os.path.join(current_save_path,"data.npz"),x=feature,y=np.where(index==label[j])[0][0])
                    counter[label[j]] = counter[label[j]] + 1
                    print(f"{j}_{file_name}-done")
                print(f"finished the {label[j]} label")


if __name__ == '__main__':
    os.chdir("D:\TL\V1")
    conf=Config()
    SAVE_PATH = conf.save_path
    MODES = conf.mode  # MODES: "maintenance" | "dynamic"
    FS=conf.fs
    SEGMENTATION_LENGTH= conf.segmentation_length
    OVERLAP_RATIO=conf.overlap_ratio
    DISCARD_LENGTH=conf.discard_length
    FFT_SIZE = 256
    subj_list=np.arange(20)
    # gen_training_data(0)
    # gen_training_data(0)
    for subj_id in subj_list:
        # proc=Process(target=gen_training_data,args=(subj_id,))
        # proc.start()
        gen_training_data(subj_id)
