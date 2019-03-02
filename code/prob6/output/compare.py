import numpy as np

def data(filename):
    q = []
    with open(filename) as txt_file:
        for line in txt_file:
            q.append(int(line.strip()))
    return q

q1 = data('predictions_q1.txt')
q1 = np.array(q1)
q2 = data('predictions_q2.txt')
q2 = np.array(q2)
q3 = data('predictions_q3.txt')
q3 = np.array(q3)
q4 = data('predictions_q4.txt')
q4 = np.array(q4)
q5 = data('predictions_q5.txt')
q5 = np.array(q5)

print((q1==q2).sum()/len(q1))
print((q2==q3).sum()/len(q1))
print((q3==q4).sum()/len(q1))
print((q4==q5).sum()/len(q1))