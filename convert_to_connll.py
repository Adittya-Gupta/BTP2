

with open('output.txt', 'r') as f:
    with open('train.conll', 'w') as f2:
        last = 'O'
        for line in f:
            if line == '\n':
                f2.write('\n')
                last = 'O'
            else:
                flg = False if line.split()[1] == last else True
                f2.write(line.split()[0] + '\t' + ('O' if line.split()[1] == 'O' else ('B-' if flg else 'I-') + line.split()[1].split('-', 1)[1]) + '\n')
                last = line.split()[1]
        