from multiprocessing.connection import Client
from time import sleep

address = ('localhost', 12345)

with Client(address, authkey=b'pyphase_live') as conn:
    for i in range(200):
        conn.send([i+1,i+2,i+3,i+4,i+5,i**2,i**3,i**0.1,i**0.2,i**0.5])
    conn.send('quit')