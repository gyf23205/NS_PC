# This file define some hyperparameters of the decision network
def init(hypers):
    global n_embed_channel
    global n_rel
    global discount
    n_embed_channel, n_rel, discount = hypers[0], hypers[1], hypers[2]