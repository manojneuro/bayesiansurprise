import tqdm
import numpy as np

def lcs(x, y):
    '''
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    '''
    m = len(x)
    n = len(y)
    c = np.zeros((m+1,n+1), dtype=np.int)

    for i in tqdm.trange(1,m+1, leave=True, desc='Aligning'):
        for j in range(1,n+1):
            if x[i-1] == y[j-1]:
                c[i,j] = c[i-1,j-1] + 1
            else:
                c[i,j] = max(c[i,j-1], c[i-1,j])

    mask1, mask2 = [], []
    i = m 
    j = n 
    while i > 0 and j > 0: 
      if x[i-1] == y[j-1]: 
        i-=1
        j-=1
        mask1.append(i)
        mask2.append(j)

      elif c[i-1][j] > c[i][j-1]: 
        i-=1
      else: 
        j-=1

    return mask1[::-1], mask2[::-1]
