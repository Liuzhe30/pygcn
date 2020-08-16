import numpy as np

content = np.loadtxt('/data/rasa.content')
with open('/data/join.fasta') as fasta:
    with open('/data/rasa.cites', 'w') as w:
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