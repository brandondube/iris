from multiprocessing.connection import Client
from time import sleep

import numpy as np

import dill as pickle

def cost_function(xk):
    return np.asarray(xk).sum()

address = ('localhost', 12345)

with Client(address, authkey=b'pyphase_live') as conn:
    i=0
    arr = [i+1,i+2,i+3,i+4,i+5,i**2,i**3,i**0.1,i**0.2,i**0.5]

    # with open('stupid.pkl', 'wb') as fid:
    #     pickle.dump(cost_function, fid)
    
    # with open('stupid.pkl', 'rb') as fid:
    fcn = pickle.dumps(cost_function)
    #conn.send(arr)
    conn.send({
        'id': 'name_values',
        'data': [str(x) for x in range(len(arr))],
    })
    print(cost_function)
    conn.send({
        'id': 'add_cost_function',
        'data': fcn,
    })
    for i in range(200):
        conn.send([i+1,i+2,i+3,i+4,i+5,i**0.1,i**0.5, 2*i])
    conn.send('quit')
