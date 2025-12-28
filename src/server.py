import numpy as np

def fed_avg(global_weights, client_weights_list):
    # Create empty list of zeros matching structure
    new_weights = [np.zeros_like(w) for w in global_weights]
    num_clients = len(client_weights_list)
    
    # Sum
    for client_w in client_weights_list:
        for i in range(len(new_weights)):
            new_weights[i] += client_w[i]
            
    # Average
    new_weights = [w / num_clients for w in new_weights]
    return new_weights