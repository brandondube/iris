from multiprocessing.connection import Listener

from matplotlib import pyplot as plt

import dill as pickle

from pyphase.plotpanels import ConvergencePanel
plt.style.use('ggplot')

address = ('localhost', 12345)     # family is deduced to be 'AF_INET'
with Listener(address, authkey=b'pyphase_live') as listener:
    with listener.accept() as conn:
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
