from multiprocessing.connection import Listener

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import dill as pickle

from pyphase.plotpanels import ConvergencePanel


# import stuff to allow a cost function to be executed
from prysm import (
    FringeZernike,
    Seidel,
    PSF,
    MTF,
    mtf_tan_sag_to_dataframe,
)
cost_function = None
address = ('localhost', 12345)     # family is deduced to be 'AF_INET'
with Listener(address, authkey=b'pyphase_live') as listener:
    print('pyphase plotting server started')
    with listener.accept() as conn:
        print('client connected')
        # make a new panel instance
        panel = ConvergencePanel()
        while True:
            msg = conn.recv()
            if msg == 'quit':
                plt.ioff()
                plt.show()
                break

            if type(msg) is dict:
                # dicts are used to send plot modification messages.
                # valid messages have the signature:
                #   {
                #       'id': abcd,
                #       'data': wxyz,
                #   }
                # idopts = {
                #    'name_values': None,
                #    'final_result': None,
                #    'add_cost_function': None,
                # }
                mid = msg['id'].lower()
                if mid == 'name_values':
                    panel.add_labels(msg['data'])
                    continue

                if mid == 'add_cost_function':
                    panel.add_cost_function(pickle.loads(msg['data']))
                    continue

                if mid == 'final_result':
                    # TODO: implement
                    continue
            
            panel.current_data = msg
