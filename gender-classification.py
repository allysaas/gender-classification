#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:32:38 2020

@author: allysaas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:33:55 2019

@author: allysaas
"""
import random
import pandas as pd
import csv
import numpy as np
import math
import itertools

def get_from_csv(filename):
    data_voice = []
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        voice_target = []
        voice_filename = []
        for row in csv_reader:
            voice = []
            qq = 0
            for number in row:
                if number == 'class':
                    voice_target.append(row[number])
                elif number == 'filename':
                    voice_filename.append(row[number])
                else:
                    voice.append(float(row[number]))
                    qq += 1
            data_voice.append(voice)   
    #print(list_voice)
    return data_voice, voice_target, voice_filename



def split_dataset(data_voice, voice_target, voice_filename):
    data_voice_M = []
    voice_filename_M = []
    voice_target_M = []
    data_voice_F = []
    voice_filename_F = []
    voice_target_F = []
    
    for i, kr in enumerate(voice_target):
        if kr == 'm':
            data_voice_M.append(data_voice[i])
            voice_filename_M.append(voice_filename[i])
            voice_target_M.append(kr)
        elif kr == 'f':
            data_voice_F.append(data_voice[i])
            voice_filename_F.append(voice_filename[i])
            voice_target_F.append(kr)
    #result = [item for sublist in zip(contoh_1,contoh_2) for item in sublist]
    data_voice_train = [item for sublist in zip(data_voice_F[:50], data_voice_M[:50]) for item in sublist]
    #data_voice_train = data_voice_F[:50] + data_voice_M[:50]
    voice_target_train = [item for sublist in zip(voice_target_F[:50], voice_target_M[:50]) for item in sublist]
    #voice_target_train = voice_target_F[:50] + voice_target_M[:50]
    voice_filename_train = [item for sublist in zip(voice_filename_F[:50], voice_filename_M[:50]) for item in sublist]
    #voice_filename_train = voice_filename_F[:50] + voice_filename_M[:50]
    
    data_voice_test = [item for sublist in zip(data_voice_F[50:58], data_voice_M[50:58]) for item in sublist]
    #data_voice_test = data_voice_F[50:58] + data_voice_M[50:58]
    voice_target_test = [item for sublist in zip(voice_target_F[50:58], voice_target_F[50:58]) for item in sublist]
    #voice_target_test = voice_target_F[50:58] + voice_target_M[50:58]
    voice_filename_test = [item for sublist in zip(voice_filename_F[50:58], voice_filename_M[50:58]) for item in sublist]
    #voice_filename_test = voice_filename_F[50:58] + voice_filename_M[50:58]
    
    #print(len(data_voice_F)+len(data_voice_F))
    
    return data_voice_train, voice_target_train, voice_filename_train, data_voice_test, voice_target_test, voice_filename_test

#print(split_dataset(data_voice, voice_target, voice_filename))

frame_len = 399
    
#print(voice_filename_test)
#print(voice_target_test)
def lvq_train(learning_rate, X, target, threshold):
    nn = np.unique(target)
    bobot_index = []
    for i, items in enumerate(nn):
        bobot_index.append(random.choice(np.where(target == items)[0]))
    bobot_index = [19, 67]
    print(bobot_index)
    print(X[67])
    print("=========================== BOBOT AWAL ============================")
    bobot = X[bobot_index].astype(np.float64)
    print(bobot) 
    print("========================== BOBOT AKHIR ============================")
    bobot_class = target[bobot_index]
    X = np.delete(X, [index for index in bobot_index], 0)
    target = np.delete(target, [index for index in bobot_index], 0)
    stop = 0
    print(len(X))
    while (stop != threshold):
        for i, item in enumerate(X):
            new = []
            for j, weight in enumerate(bobot):
                # euclidean distance
                #new.append(math.sqrt(sum(abs(weight-item)**2)))
                # normalized cross correlation
                new.append(1/frame_len*(sum(weight*item)/(sum(weight)*sum(item))))
                #new.append(sum(weight*item)/math.sqrt(sum(weight**2)*sum(item**2)))
                # manhattan distance
                #new.append(sum(abs(weight-item)))
                # cosine similarity
                #new.append(sum(weight*item)/((math.sqrt(sum(weight**2)))*(math.sqrt(sum(item**2)))))
            target_min = new.index(min(new))
            if (target[i] == bobot_class[target_min]):
                bobot[target_min] = (bobot[target_min] + ((item-bobot[target_min])*learning_rate)).tolist()
            else:
                bobot[target_min] = (bobot[target_min] - ((item-bobot[target_min])*learning_rate)).tolist()
        stop += 1
        learning_rate /= 2
    print(bobot)
    return bobot, bobot_class

#bobot, bobot_class, bobot_index = lvq_train(1, np.array(data_voice_train), np.array(voice_target_train), 10)
#print(bobot)

def lvq_test(test_data, test_target, bobot, bobot_class):
    bbt_class = bobot_class
    #print(bobot)
    target_class = {}
    right_class = 0
    print("====================== Kelas sblmnya ======================")
    print(*test_target)
    for i, items in enumerate(test_data):
        hasil = []
        for j, weight in enumerate(bobot):
            # euclidean distance
            #hasil.append(math.sqrt(sum(abs(weight-items)**2)))
            # normalized cross correlation
            hasil.append(1/frame_len*(sum(weight*items)/(sum(weight)*sum(items))))
            #hasil.append(sum(weight*items)/(math.sqrt((sum(np.power(weight,2))*sum(np.power(items,2))))))
            # manhattan distance
            #hasil.append(sum(abs(weight-items)))
            # cosine similarity
            #hasil.append(sum(weight*items)/((math.sqrt(sum(np.power(weight,2))))*(math.sqrt(sum(np.power(items,2))))))
        target_class[i] = bbt_class[hasil.index(min(hasil))]
        if target_class[i] == test_target[i]:
            right_class += 1
    print("====================== HASIL ======================")
    print(*target_class.items(), sep='\n')
    print("============================= AKURASI ==============================")
    accuracy = right_class/len(test_data)
    print(accuracy)
    return accuracy

#lvq_test(data_voice_test, voice_target_test, bobot)

#def evaluasi_bobotawal():
#    best_accuracy = 0
#    best_bobot = []
#    for i in range(500):
#        bobot, bobot_class = lvq_train(0.1, np.array(data_voice_train), np.array(voice_target_train), 250)
#        accuracy = lvq_test(data_voice_test, voice_target_test, bobot)
#        if (accuracy > best_accuracy):
#            best_accuracy = accuracy
#            best_bobot = bobot_index
#    print("=================== BEST AKURASI ==================")
#    print(best_accuracy)
#    print("=================== BEST BOBOT ====================")
#    print(best_bobot)
#    print([np.array(data_voice_train)[kr] for i,kr in enumerate(best_bobot)])
        
#evaluasi_bobotawal() 
        

def cross_validation_split(dataset, dataset_target, dataset_name, n_folds):
    fold_size = int(len(dataset)/n_folds)
    dataset_splitted= list()
    dataset_target_splitted= list()
    dataset_name_splitted = list()
    for i in range(n_folds):
        dataset_splitted.append(dataset[0:fold_size])
        dataset = np.delete(dataset, [index for index in range(0, fold_size)], 0)
        
        dataset_target_splitted.append(dataset_target[0:fold_size])
        dataset_target = np.delete(dataset_target, [index for index in range(0, fold_size)], 0)
        
        dataset_name_splitted.append(dataset_name[0:fold_size])
        dataset_name = np.delete(dataset_name, [index for index in range(0, fold_size)], 0)
    #print(dataset_target_splitted)
    return dataset_splitted, dataset_target_splitted,dataset_name_splitted
        
    
#data_voice_train, voice_target_train, voice_filename_train, data_voice_test, voice_target_test, voice_filename_test
#cross_validation_split(all_data, all_data_target, all_data_voicename, 5)
#print(*dataset_split)
#TEST K FOLD CROSS VALIDATION

#print(data_test)
#print(data_target_test)
#print("len data_train: "+str(len(data_train)))
#print(data_target_train)
#flat_train = list(itertools.chain.from_iterable(data_train))
#flat_target_train = list(itertools.chain.from_iterable(data_target_train))
#flat_test = list(itertools.chain.from_iterable(data_test))
#flat_target_test = list(itertools.chain.from_iterable(data_target_test))
def main():
    #get dataset from excel
    data_voice, voice_target, voice_filename = get_from_csv('Dataset.csv')
    data_voice_train, voice_target_train, voice_filename_train, data_voice_test, voice_target_test, voice_filename_test = split_dataset(data_voice, voice_target, voice_filename)
    learning_rate = 0.1
    
    #gabung seluruh data train dan test
    all_data = data_voice_train + data_voice_test
    all_data_target = voice_target_train + voice_target_test
    all_data_voicename = voice_filename_train + voice_filename_test
    
    #k fold cross validation
    dataset_split, dataset_target_split, dataset_name_split = cross_validation_split(all_data, all_data_target, all_data_voicename, 5)
    data_train = list(itertools.chain.from_iterable(dataset_split[0:1])) + list(itertools.chain.from_iterable(dataset_split[2:5]))
    data_test = dataset_split[1]
    data_target_train = list(itertools.chain.from_iterable(dataset_target_split[0:1]))  + list(itertools.chain.from_iterable(dataset_target_split[2:5])) 
    data_target_test = dataset_target_split[1]
    bobot, bobot_class = lvq_train(0.1, np.array(data_train), np.array(data_target_train), 100)
    lvq_test(data_test, data_target_test, bobot, bobot_class)

main()