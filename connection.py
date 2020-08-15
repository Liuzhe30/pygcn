import numpy as np

content = np.loadtxt('/home/liuz/8_pytorch_gcn/data/rasa.content')
with open('/home/liuz/8_pytorch_gcn/data/join.fasta') as fasta:
    with open('/home/liuz/8_pytorch_gcn/data/rasa.cites', 'w') as w:
        line = fasta.readline()
        num = 0
        while line:
            if(line[0] == '>'):
                line = fasta.readline()
                continue
            length = len(line.strip())
            
            for i in range(0,length-1):
                w.write(str(int(content[i+num][0])) + " " + str(int(content[i+num+1][0])) + '\n')
            
            num += length
            line = fasta.readline()        