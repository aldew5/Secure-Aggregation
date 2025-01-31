from flask import *
from flask_socketio import SocketIO, emit
from flask_socketio import *
from utils import *
import pickle
import numpy as np
import random
import os
import time, datetime
from train import *
from model import *

# global variables
server_port = 2019
num_clients = 5
threshold = 5
dim = (1, 12)
lr = 1e-4
num_epochs = 100
batch_size = 48
X_test, y_test = get_test_data()
rsquare_thres = 0.001

# NOTE:
# Does not implement round 3 (consistency checks) which is not
# required in the honest-but-curious setting


def sleep_for_a_while(s):
    # print(f"### {s}: sleeping for 5 seconds")
    time.sleep(5)
    # print(f"### {s}: woke up")

class secaggserver:
    def __init__(self, port, dim=dim, n=4, t=3):
        self.port = port
        self.dim = dim
        self.n = n  # the number of users
        self.t = t  # the threshold
        self.iter_no = 0
        self.model_weights = np.zeros(self.dim)
        self.start_time = None
        self.end_time = None
        self.initial_time = datetime.datetime.now()

        self.all_seen_clients = set()

        self.mses = []
        self.rsquares = []
        self.timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]
        self.out_dir = os.path.join("output", self.timestamp)
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)

        self.clear()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()

    def clear(self):
        """
        Called between each iteration
        
        Resets key dictionaries and updates round and iteration counters
        """
        self.U_0, self.U_1, self.U_2, self.U_3, self.U_4 = [], [], [], [], []
        self.ready_client_ids = set()
        
        # private keys
        self.c_pk_dict, self.s_pk_dict, self.e_uv_dict = {}, {}, {}
        self.sk_shares_dict, self.b_shares_dict = {}, {}

        self.aggregated_value = np.zeros(self.dim)

        self.lasttime = time.time()
        self.curr_round = -1
        self.iter_no += 1
    
    #TODO: can be combined with above
    def move_to_next_iteration(self):
        # emit message to client
        for client_id in self.ready_client_ids:
            emit("waitandtry", room=client_id) 
        self.clear()

    def gen_mask(self, seed):
        """
        Use a seed to generate a random mask with same shape as the input
        """
        np.random.seed(seed)
        return np.float64(np.random.rand(self.dim[0], self.dim[1]))

    def meta_handler(self, name, roundno, userlist, resp, add_info):
        """
        add_info: add_info function corresponding to the particular round
            
        Handles
        """
        if (self.curr_round != roundno) or request.sid not in userlist:
            emit('waitandtry')
        else:  # Add entries for punctual users
            # actually in round, or this is the first time we pass the limit
            if (time.time()-self.lasttime > time_max):
                # print(f"Arrived too late for Round {roundno}: {name}")
                emit('waitandtry')
            else:
                # print(f"{request.sid} sends info for Round {roundno}")
                add_info(resp)
                
    # Advertise keys
    # - user sends signed public keys
    def round_0_add_info(self, resp):
        self.ready_client_ids.add(request.sid)
        public_keys = pickle.loads(resp)
        # c_u sk, pk are DH keypairs
        self.c_pk_dict[request.sid] = public_keys['c_u_pk']
        self.s_pk_dict[request.sid] = public_keys['s_u_pk']


    def round_0_attempt_action(self):
        """either start next round, move to next iteration, or keep on waiting for next client (default fall through)"""
        #print("START", datetime.datetime.now())
        self.start_time = datetime.datetime.now()
        if (self.curr_round != 0): return
    
        # start next round
        # either we have all clients or at least as many as the min threshold
        if (len(self.ready_client_ids) == self.n) or \
            (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
            
            # notify of second case
            if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                print("NOTE: max time has passed without receiving response from all clients, but enough clients have responded")
                
            self.curr_round = 1
            print(f"Collected keys from {len(self.ready_client_ids)} clients -- Starting Round 1.")
            
            ready_clients = list(self.ready_client_ids)
            self.U_1 = ready_clients

            self.ready_client_ids.clear()
            self.lasttime = time.time()
            for client_id in ready_clients:
                # dumps serializes object hierarchy
                emit('share_keys', (pickle.dumps(self.c_pk_dict), pickle.dumps(self.s_pk_dict)), room=client_id)
            
            time.sleep(time_max)
            # next round
            if (self.curr_round == 1):
                self.round_1_attempt_action() # in case someone disconnects

        # move to next iteration
        elif (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
            self.move_to_next_iteration()
        # do nothing (wait for next client)

    # Share keys
    # - Server must broadcast list of received public keys to all users
    # - User validates signatures, computing s_{u, v}
    # - User computes t-out-of-n secret shares and sends encrypted verions to server
    def round_1_add_info(self, resp):
        self.ready_client_ids.add(request.sid)
        e_uv_dict = pickle.loads(resp)
        # for every user v, record dict[v][u] = e_{u,v}
        for key, value in e_uv_dict.items():
            if key not in self.e_uv_dict:
                self.e_uv_dict[key] = {}
            self.e_uv_dict[key][request.sid] = value

    def round_1_attempt_action(self):  # ShareKeys
        """
        Either start next round, move to next iteration, or keep on waiting
        for next client (default fall through)
        """
        if (self.curr_round != 1):
            return

        # start next round
        if len(self.ready_client_ids) == len(self.U_1) or \
                (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
            if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                print("NOTE: max time has passed without receiving response from all clients, but enough clients have responded")
            self.curr_round = 2
            
            # received validate signatures
            print(f"Collected e_uv from {len(self.ready_client_ids)} clients -- Starting Round 2.")

            ready_clients = list(self.ready_client_ids)
            self.U_2 = ready_clients
            
            self.ready_client_ids.clear()
            for client_id in ready_clients:
                # send signal for the next round
                emit('masked_input_collection', pickle.dumps(
                    self.e_uv_dict[client_id]), room=client_id)
            self.lasttime = time.time()
            
            time.sleep(time_max)
            # only called if round 2 action never succeeded
            if (self.curr_round == 2):
                # in case someone disconnects
                self.round_2_attempt_action()

        # move to next iteration
        elif (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
            self.move_to_next_iteration()
        # do nothing (wait for next client)

    # MaskedInputCollection
    # - Server forwards received encrypted shares
    # - User computes maksed input y_u and sends back to server
    def round_2_add_info(self, resp):
        # decode the masked input
        self.ready_client_ids.add(request.sid)
        masked_input = pickle.loads(resp)
        # these inputs are gradients?
        self.aggregated_value += masked_input

    def round_2_attempt_action(self):
        """either start next round, move to next iteration, or keep on waiting for next client (default fall through)"""
        if (self.curr_round != 2):
            return

        # start next round
        if len(self.ready_client_ids) == len(self.U_2) or \
                (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
            if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                print("NOTE: max time has passed without receiving response from all clients, but enough clients have responded")
            self.curr_round = 3
            print(f"Collected y_u from {len(self.ready_client_ids)} clients -- Starting Round 3.")

            ready_clients = list(self.ready_client_ids)
            self.U_3 = ready_clients
            
            self.ready_client_ids.clear()
            for client_id in ready_clients:
                emit('unmasking', pickle.dumps(self.U_3), room=client_id)
            self.lasttime = time.time()
            
            time.sleep(time_max)
            if (self.curr_round == 3):  # only called if round 3 action never succeeded
                self.round_3_attempt_action()  # in case someone disconnects

        # move to next iteration
        elif (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
            self.move_to_next_iteration()
        # do nothing (wait for next client)
    
    # Unmasking (skip consistency check)
    # - Server sends a list {v, sigma'} of survived users
    # - User sends shares of b_u for aliver users and s_u^{pk} for dropped
    # - Then server can reconstruct the final aggregated value
    def round_3_add_info(self, resp):  # Unmasking
        # decode the masked input
        self.ready_client_ids.add(request.sid)
        data = pickle.loads(resp)
        for id, share in data[0].items():
            if id not in self.sk_shares_dict:
                self.sk_shares_dict[id] = []
            self.sk_shares_dict[id].append(share)
        for id, share in data[1].items():
            if id not in self.b_shares_dict:
                self.b_shares_dict[id] = []
            self.b_shares_dict[id].append(share)

    def round_3_attempt_action(self):  # Unmasking
        if (self.curr_round != 3):
            return

        # compute result
        if len(self.ready_client_ids) == len(self.U_3) or \
                (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
            if (time.time()-self.lasttime > time_max and len(self.ready_client_ids) >= self.t):
                print("NOTE: max time has passed without receiving response from all clients, but enough clients have responded")
            self.curr_round = 4  # even though no more rounds after it
            print(f"Collected y_u from {len(self.ready_client_ids)} clients -- Starting Round 4 (final round).")

            ready_clients = list(self.ready_client_ids)
            self.U_4 = ready_clients
            # print("U_4", ready_clients)
            mask = np.zeros(self.dim)

            # reconstruct s_{u,v} for all u in U2\U3
            dropped_out_users = [u for u in self.U_2 if u not in self.U_3]
            for u in dropped_out_users:
                # reconstruct secrets
                s_u_sk = SS.recon(self.sk_shares_dict[u])

                for v in self.U_3:
                    sv_pk = self.s_pk_dict[v]
                    shared_key = KA.agree(s_u_sk, sv_pk)
                    random.seed(shared_key)
                    s_uv = random.randint(0, 2**32 - 1)
                    if v > u:
                        mask += self.gen_mask(s_uv)
                    elif v < u:
                        mask -= self.gen_mask(s_uv)

            # reconstruct b_u for all u in U3
            for u in self.U_3:
                b_u = SS.recon(self.b_shares_dict[u])
                mask -= self.gen_mask(b_u)

            # subtract the mask to get the final value
            self.aggregated_value += mask
            
            # update the model weights using the average of all clients' gradients
            self.model_weights += self.aggregated_value / len(self.U_3)

            # show the R^2 and MSE result
            LR = LinearRegression(lr, num_epochs, batch_size, self.model_weights)
            mse, rsquare = LR.eval(X_test, y_test)
            print('Test R^2: ' + str(rsquare))
            print("Test MSE: " + str(mse))

            self.mses.append(mse)
            self.rsquares.append(rsquare)
           
            if self.iter_no >= 10:
                # also, R^2 sufficiently low
                if abs(self.rsquares[-2]-self.rsquares[-3]) < rsquare_thres and  \
                   abs(self.rsquares[-1]-self.rsquares[-2]) < rsquare_thres:
                    print("Ready to finish training")
                    
                    print("TRAIN START:", self.initial_time)
                    print("TRAIN FINISH", datetime.datetime.now())
                    
                    #plot(self.out_dir, np.array(self.mses), np.array(self.rsquares))
                    for client_id in self.all_seen_clients: emit("disconnect", room=client_id)
                    return
            
                #if (self.iter_no-1) % 10 == 0:
                #plot(self.out_dir, np.array(self.mses), np.array(self.rsquares))
            
            # emits weight and try
            #print("END", datetime.datetime.now())
            self.end_time = datetime.datetime.now()
            print(self.end_time - self.start_time)
            #self.total_time += (self.end_time - self.start_time).microseconds
            #print(self.total_time)
            self.move_to_next_iteration()

        # move to next iteration
        elif (time.time()-self.lasttime > time_max and len(self.ready_client_ids) < self.t):
            print("could not compute final aggregate")
            self.move_to_next_iteration()




    def register_handles(self):
        # when new client connects or exisitng client renotify server of their alive state
        @self.socketio.on("connect")
        def handle_connect():
            self.all_seen_clients.add(request.sid)

            if(self.curr_round != -1):
                # print("Protocol has already begun: wait and try")
                emit("waitandtry")
            else: # protocol has not started
                print(request.sid, " Connected")
                self.ready_client_ids.add(request.sid)
                if len(self.ready_client_ids) == self.n:
                    print('\n' + '-' * 20 + "iteration " + str(self.iter_no) + '-' * 20 )
                    print("All clients connected -- Starting Round 0.")
                    
                    ready_clients = list(self.ready_client_ids)
                    self.U_0 = ready_clients
                    self.ready_client_ids.clear()
                    self.curr_round = 0
                    for client_id in ready_clients:
                        emit("advertise_keys_and_train_model", (pickle.dumps(client_id), pickle.dumps(self.model_weights)), room=client_id)
                    self.lasttime = time.time()
        @self.socketio.on("retryconnect")
        def handle_retryconnect():
            handle_connect()

        # Round 0 (AdvertiseKeysAndTrainModel)
        @self.socketio.on('done_advertise_keys_and_train_model')
        def handle_advertise_keys_and_train_model(resp):
            # also send model weights
            self.meta_handler("AdvertiseKeysAndTrainModel", 0, self.U_0, resp, self.round_0_add_info)
            self.round_0_attempt_action()

        # Round 1 (ShareKeys)
        @self.socketio.on('done_share_keys')
        def handle_share_keys(resp):
            self.meta_handler("round1-ShareKeys", 1, self.U_1, resp, self.round_1_add_info)
            self.round_1_attempt_action()
                
        # Round 2 (MaskedInputCollection)
        @self.socketio.on('done_masked_input_collection')
        def handle_masked_input_collection(resp):
            self.meta_handler("round2-MaskedInputCollection", 2, self.U_2, resp, self.round_2_add_info)
            self.round_2_attempt_action()

        # Round 3 (Unmasking)
        @self.socketio.on('done_unmasking')
        def handle_unmasking(resp):
            self.meta_handler("round3-Unmasking", 3, self.U_3, resp, self.round_3_add_info)
            self.round_3_attempt_action()


        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
                self.all_seen_clients.remove(request.sid)

    def start(self):
        self.socketio.run(self.app, port=self.port)

if __name__ == '__main__':
    time_max = 5
    server = secaggserver(server_port, dim=dim, n=num_clients, t=threshold)
    # print("listening on http://localhost:2019")
    server.start()
