import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import csv

from third_octave import ThirdOctaveTransform

class PresPredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, train_data_path=None, sampling_rate=32000, val_split=0):
        self.train_data_path = train_data_path
        self.sr = sampling_rate
        self.val_split = val_split
        
        self.tob_transform = ThirdOctaveTransform()

        if not os.path.isfile(dataset_file + ".npz"):
            assert train_data_path is not None, "no location for dataset files specified"
            self.create_dataset(train_data_path, dataset_file)
        else:
            self.dataset_file = dataset_file

        train_data = np.load(self.dataset_file + ".npz", mmap_mode='r')
        self.train_audio_data = train_data["arr_0"]
        self.train_pres_data = train_data["arr_1"]
        if self.val_split != 0:
            val_data = np.load(self.dataset_file + "_val.npz", mmap_mode='r')
            self.val_audio_data = val_data["arr_0"]
            self.val_pres_data = val_data["arr_1"]

        self.train = 1
        print(self.__len__())
        self.train = 0
        print(self.__len__())

    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files, pres_files = list_all_audio_files(location + 'sound/', location + 'pres_profile')
        processed_files = []
        f_len = 32768
        x_tob_train = []
        x_pres_train = []
        if self.val_split!=0:
            x_tob_val = []
            x_pres_val = []
        for i, file in enumerate(files):
            print(" -> Processed " + str(i) + " of " + str(len(files)) + " files")
            print(pres_files[i])
            x, _ = lr.load(path=file, sr=self.sr, mono=True)
            n_f = int(np.floor(x.shape[0]/f_len))
            
            # Presence profile
            with open(pres_files[i]+'_pp2_T.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    x_pres_T = [[float(l) for l in row]]
                if len(x_pres_T[0]) != n_f:
                    for indf in range(n_f-len(x_pres_T[0])):
                        x_pres_T[0].append(float(0))
            with open(pres_files[i]+'_pp2_V.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    x_pres_V = [[float(l) for l in row]]
                if len(x_pres_V[0]) != n_f:
                    for indf in range(n_f-len(x_pres_V[0])):
                        x_pres_V[0].append(float(0))
            with open(pres_files[i]+'_pp2_B.csv', newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    x_pres_B = [[float(l) for l in row]]
                if len(x_pres_B[0]) != n_f:
                    for indf in range(n_f-len(x_pres_B[0])):
                        x_pres_B[0].append(float(0))
            x_pres_temp = np.concatenate((x_pres_T, x_pres_V, x_pres_B))
            if self.val_split==0 or i<(1-self.val_split)*len(files):
                for indf in range(n_f):
                    x_tob_train.append(self.tob_transform.wave_to_third_octave(x[indf*f_len:(indf+1)*f_len]))
                    x_pres_train.append(x_pres_temp[:, indf])
            else:
                for indf in range(n_f):
                    x_tob_val.append(self.tob_transform.wave_to_third_octave(x[indf*f_len:(indf+1)*f_len]))
                    x_pres_val.append(x_pres_temp[:, indf])

        np.savez(self.dataset_file + ".npz", x_tob_train, x_pres_train)
        if self.val_split!=0:
            np.savez(self.dataset_file + "_val.npz", x_tob_val, x_pres_val)
            

    def __getitem__(self, idx):
        if self.train == 1:
            input_x = torch.unsqueeze(torch.from_numpy(self.train_audio_data[idx]), 0)
            pres = torch.unsqueeze(torch.from_numpy(self.train_pres_data[idx]), 0)
        else:
            input_x = torch.unsqueeze(torch.from_numpy(self.val_audio_data[idx]), 0)
            pres = torch.unsqueeze(torch.from_numpy(self.val_pres_data[idx]), 0)
        return input_x, pres
        
    def __len__(self):
        if self.train == 1:
            return len(self.train_audio_data)
        else:
            if self.val_split != 0:
                return len(self.val_audio_data)
            else:
                return 0

def list_all_audio_files(audio_location, pres_location=None):
    pres_files = []
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(audio_location):
        for filename in [f for f in sorted(filenames) if f.endswith((".mp3", ".wav", ".aif", "aiff")) and "channel" not in f]:
            pres_files.append(os.path.join(pres_location, filename[:-4]))
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + audio_location)
    return audio_files, pres_files




# No ground truth data on presence, test only
class PresPredDatasetSimple(torch.utils.data.Dataset):
    def __init__(self, dataset_file, train_data_path=None, sampling_rate=32000):
        self.train_data_path = train_data_path
        self.sr = sampling_rate
        
        self.tob_transform = ThirdOctaveTransform()

        if not os.path.isfile(dataset_file + ".npz"):
            assert train_data_path is not None, "no location for dataset files specified"
            self.create_dataset(train_data_path, dataset_file)
        else:
            self.dataset_file = dataset_file

        train_data = np.load(self.dataset_file + ".npz", mmap_mode='r')
        self.train_audio_data = train_data["arr_0"]

    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files = list_all_audio_files_simple(location + 'mix/')
        processed_files = []
        f_len = 32768
        x_tob_train = []
        for i, file in enumerate(files):
            print(" -> Processed " + str(i) + " of " + str(len(files)) + " files")
            print(files[i])
            x, _ = lr.load(path=file, sr=self.sr, mono=True)
            n_f = int(np.floor(x.shape[0]/f_len))
            
            for indf in range(n_f):
                x_tob_train.append(self.tob_transform.wave_to_third_octave(x[indf*f_len:(indf+1)*f_len]))
        np.savez(self.dataset_file + ".npz", x_tob_train)

    def __getitem__(self, idx):
        input_x = torch.unsqueeze(torch.from_numpy(self.train_audio_data[idx]), 0)
        return input_x
        
    def __len__(self):
        return len(self.train_audio_data)

def list_all_audio_files_simple(audio_location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(audio_location):
        for filename in [f for f in sorted(filenames) if f.endswith((".mp3", ".wav", ".aif", "aiff")) and "channel" not in f]:
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + audio_location)
    return audio_files
