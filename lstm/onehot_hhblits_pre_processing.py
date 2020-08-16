import numpy as np
import os
from keras.utils import to_categorical
import math

def format(f, n):
    if round(f) == f:
        m = len(str(f)) - 1 - n
        if f / (10 ** m) == 0.0:
            return f
        else:
            return float(int(f) / (10 ** m) * (10 ** m))
    return round(f, n - len(str(int(f)))) if len(str(f)) > n+1 else f

dict = {'C':0, 'D':1, 'S':2, 'Q':3, 'K':4,
        'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
        'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
        'A':15, 'V':16, 'E':17, 'Y':18, 'M':19}
        
# convert back map meaning; 0 for Outer-side, 1 for TM-Helix, 2 for Inner-side
topo_dict = {0: 'O', 1: 'M', 2: 'I'}

ss_dict8 = {'H':0, 'B':1, 'E':2, 'G':3,
           'I':4, 'T':5, 'S':6, '$':7}

ss_dict3 = {'H':0, 'B':1, 'E':1, 'G':0,
           'I':0, 'T':2, 'S':2, '$':2}


class Processor:
    
    def data_pre_processing(self, fasta_path, rasa_path, window_length, tag):
        train_fasta = open(fasta_path)
        line = train_fasta.readline()
        pdb_id = ""
        x_train = []
        ss_train = []
        topo_train = []
        rasa_train = []
        while line:
            codelist = []
            if(line[0] == ">"):
                pdb_id = line[1:7]
                line = train_fasta.readline()
                continue
            #print(pdb_id)
            seq_length = len(line) - 1
            
            '''
            #-------- used for feature without one-hot code ----------#
            
            t = int((window_length - 1) / 2)
            code = np.zeros(seq_length, int)
            code = np.r_[np.ones(t, int), code]        
            code = np.r_[code, np.ones(t, int)]            
            
            #-------- used for feature without one-hot code ----------#
            '''
            
            #-------- one-hot encoded (len * 20) ----------#
            #print(seq_length)
            for i in line:
                if(i != "\n"):
                    code = dict[i.upper()]
                    codelist.append(code)
            data = np.array(codelist)
            #print('Shape of data (BEFORE encode): %s' % str(data.shape))
            encoded = to_categorical(data)
            if(encoded.shape[1] < 20):
                column = np.zeros([encoded.shape[0], 20 - encoded.shape[1]], int)
                encoded = np.c_[encoded, column]
            #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
            #print(encoded)
            #print(encoded.shape)
            #-------- one-hot encoded (len * 20) ----------#
            
            #-------- noseq encoded (len + window_length - 1) * 21 ----------#
            code = encoded
            length = code.shape[0]
            #print(length)
            noSeq = np.zeros([length, 1], int)
            code = np.c_[code, noSeq]
            
            t = int((window_length - 1) / 2)
            
            code = np.r_[np.c_[np.zeros([t, 20], int), np.ones(t, int)], code]        
            code = np.r_[code, np.c_[np.zeros([t, 20], int), np.ones(t, int)]]
            #print(code.shape)
            #-------- noseq encoded (len + window_length - 1) * 21 ----------#
            
            #-------- add hhblits feature ----------#
            length = code.shape[0]
            list_dir = os.getcwd()
            hhm_path = '/home/liuz/6_SS_topo_new/newdata/icdtools_HHResults_new/' + pdb_id + '.hhm'
            hhm_file = os.path.join(list_dir, hhm_path)
            if(os.path.exists(hhm_file)):    
                with open(hhm_file) as hhm:
                    hhm_matrix = np.zeros([length, 30], float)
                    hhm_line = hhm.readline()
                    top = t - 1
                    while(hhm_line[0] != '#'):
                        hhm_line = hhm.readline()
                    for i in range(0,5):
                        hhm_line = hhm.readline()
                    #print(hhm_line)  
                    while hhm_line:
                        if(len(hhm_line.split()) == 23):
                            each_item = hhm_line.split()[2:22]
                            for idx, s in enumerate(each_item):
                                if(s == '*'):
                                    each_item[idx] = '99999'                            
                            for j in range(0, 20):
                                if(top == length - 1 - t):
                                    break
                                try:
                                    hhm_matrix[top, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))
                                except IndexError:
                                    pass
                        elif(len(hhm_line.split()) == 10):
                            each_item = hhm_line.split()[0:10]
                            for idx, s in enumerate(each_item):
                                if(s == '*'):
                                    each_item[idx] = '99999'                             
                            for j in range(20, 30):
                                if(top == length - 1 - t):
                                    break
                                try:
                                    hhm_matrix[top, j] = 10/(1 + math.exp(-1 * int(each_item[j-20])/2000))
                                except IndexError:
                                    pass                            
                            top += 1
                        hhm_line = hhm.readline()
                #print(hhm_matrix.shape)
                code = np.c_[code, hhm_matrix]
            else:
                code = np.c_[code, np.zeros([length, 30], int)]
                print(str(pdb_id) + " not found!!")                
            #-------- add hhblits feature ----------#
            
            #-------- add noseq feature ----------#
            length = code.shape[0] 
            t = int((window_length - 1) / 2)
            noSeq = np.zeros([length - window_length + 1, 1], int)
            noSeq = np.r_[np.ones([t, 1], int), noSeq]
            noSeq = np.r_[noSeq, np.ones([t, 1], int)]
            code = np.c_[code, noSeq]
            #-------- add noseq feature ----------#

            #-------- sliding window (window_length * feature) ---------#
            length = code.shape[0]
            feature = code.shape[1]
            top = 0
            buttom = window_length
            while(buttom <= length):
                #print(code[top:buttom]) 
                #print(code[top:buttom].shape) #
                x_train.append(code[top:buttom])            
                top += 1
                buttom += 1
            #print(len(window_list))
            #-------- sliding window (window_length * feature) ---------#
            
            #-------- ss mapping ---------#
            ss = open("/home/liuz/6_SS_topo_new/former/data/merge_files/" + pdb_id + ".txt")
            label = ss.readline()
            while label:
                if(label.split()[1] == 'H' or label.split()[1] == 'M'):
                    num = 0
                else:
                    num = ss_dict3[label.split()[2]]
                
                label = ss.readline()
                ss_train.append(num)
            #-------- ss mapping ---------#
            
            #-------- label mapping ---------#
            topo = open("/home/liuz/6_SS_topo_new/former/data/merge_files/" + pdb_id + ".txt")
            label = topo.readline()    
            while label:
                if(label.split()[1] == 'H' or label.split()[1] == 'M'):
                    num = 1
                else:
                    num = 0
                label = topo.readline()  
                topo_train.append(num)          
            #-------- label mapping ---------#            
            
            #-------- rasa mapping ---------#
            rasa = open(rasa_path + pdb_id + ".rasa")
            label = rasa.readline()     
            while label:
                num = float(label[0:len(label)])
                num = round(num, 3)
                if(num > 0.2):
                    rasa_train.append(1)
                else:
                    rasa_train.append(0)
                    
                #rasa_train.append(num)
                label = rasa.readline()         
            #-------- rasa mapping ---------#            
            
            line = train_fasta.readline()    
            
        #print(len(x_train))
        x_train = np.array(x_train)
        print(x_train.shape)
        #print(y_train)
        ss_train = np.array(ss_train)
        topo_train = np.array(topo_train)
        rasa_train = np.array(rasa_train)  
        print(rasa_train.shape)  
        np.save(tag + '_data/x_' + tag + '_winlen_' + str(window_length) + ".npy", x_train)   
        np.save(tag + '_data/rasa_' + tag + '_winlen_' + str(window_length) + ".npy", rasa_train)     
        
if __name__=='__main__':
    
    processor = Processor()
    
    window_length = 19
    
    tag = "test"
    fasta_path = "test_data/test.fasta"
    processor.data_pre_processing(fasta_path, '/home/baoyh/dssp_project/combined_train_rasa/', window_length, tag) 
    
    tag = "valid"
    fasta_path = "valid_data/valid.fasta"
    processor.data_pre_processing(fasta_path, '/home/baoyh/dssp_project/combined_train_rasa/', window_length, tag) 
    
    tag = "train"
    fasta_path =  "train_data/train.fasta"
    processor.data_pre_processing(fasta_path, '/home/baoyh/dssp_project/combined_train_rasa/', window_length, tag)     