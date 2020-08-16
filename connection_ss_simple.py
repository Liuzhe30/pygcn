import numpy as np

content = np.loadtxt('/home/liuz/8_pytorch_gcn/data/rasa.content')
with open('/home/liuz/8_pytorch_gcn/data/join.fasta') as fasta:
    with open('/home/liuz/8_pytorch_gcn/data/rasa.cites', 'a+') as w:
        line = fasta.readline()
        num = 0
        while line:
            if(line[0] == '>'):
                line = fasta.readline()
                continue
            length = len(line.strip())
            
            i = 0
            while i < length:
                #if(i>0 and i<length-1 and content[num+i][-2] != content[num+i-1][-2] and content[num+i][-2] != content[num+i+1][-2]):
                    #continue
                if(i == length-1 and content[num+i][-2] != content[num+i-1][-2]):
                    break
                
                flag1 = i + num
                flag2 = i + num
                for j in range(i,length-1):
                    if(content[num+j][-2] != content[num+j+1][-2]):
                        flag2 = j + num
                        break
                    if(j == length-2):
                        flag2 = length + num
                #print('i=' + str(i))
                #print('flag1=' + str(flag1))
                #print('flag2=' + str(flag2))

                w.write(str(int(content[flag1][0])) + ' ' + str(int(content[flag2][0])) + '\n')
                        #print(str(int(content[t][0])) + ' ' + str(int(content[m][0])))
                        
                if((i + flag2 - flag1 + 1) > content[-1][0]-1):
                    break
                i += flag2 - flag1 + 1
                                    
            num += length
            line = fasta.readline()