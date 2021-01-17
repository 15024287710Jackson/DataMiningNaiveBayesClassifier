import numpy as np

y=1
c=-1
a=[1,1,1,2,2,2,3,3,4]
feature=[2,3]
for v in feature:
    print(np.equal(v, a))
    t=np.sum(np.equal(v,a))
    print(t)
