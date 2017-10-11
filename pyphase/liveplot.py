from multiprocessing.connection import Listener

from matplotlib import pyplot as plt
plt.style.use('ggplot')

address = ('localhost', 12345)     # family is deduced to be 'AF_INET'
with Listener(address, authkey=b'pyphase_live') as listener:
    print('pyphase plotting server started')
    with listener.accept() as conn:
        print('client connected')
        # make a new figure for this connection
        fig, ax = plt.subplots()
        plt.ion()
        ax.set(xlabel='Generation',
               ylabel='Parameter Value')
        #plt.legend()
        #fig.canvas.manager.window.raise_()
        
        plt_labels = None
        plt_add_legend = False
        plt_data = None
        plt_lines = None
        iters = []
        iters_current = 0
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
                # }
                # this block will switch based on the value associated with id.
                mid = msg['id'].lower()
                if mid == 'name_values':
                    plt_labels = msg['data']
                    plt_add_legend = True
                    continue

                if mid == 'final_result':
                    # TODO: implement
                    continue
            
            if plt_data is None:
                # make arrays for the parameter values over generations
                plt_data = [list() for x in range(len(msg))]
                plt_lines = []

                # adapt line width to density of plot
                length = len(msg)
                line_width = 3
                if len(msg) > 5:
                    line_width = 2
                if len(msg) > 10:
                    line_width = 10
                
                # create line objects
                for idx in range(len(msg)):
                    line, = ax.plot([], [], lw=line_width)
                    plt_lines.append(line)
                
                print(len(plt_lines))
            
            if plt_add_legend:
                ax.legend(plt_labels)
            
            iters.append(iters_current)
            iters_current += 1
            for idx, val in enumerate(msg):
                plt_data[idx].append(val)
            
            for line, dat in zip(plt_lines, plt_data):
                line.set_xdata(iters)
                line.set_ydata(dat)
            
            # recompute the ax.dataLim
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            plt.pause(0.0001)
