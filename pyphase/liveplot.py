from multiprocessing.connection import Listener

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from pyphase.plotpanels import ConvergencePanel

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
                # this block will switch based on the value associated with id.
                mid = msg['id'].lower()
                if mid == 'name_values':
                    panel.add_labels(msg['data'])
                    continue

                if mid == 'add_cost_function':
                    cost_function = msg['data']

                if mid == 'final_result':
                    # TODO: implement
                    continue
            
            panel.current_data = msg
