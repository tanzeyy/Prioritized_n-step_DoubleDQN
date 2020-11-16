import time


def cur_time():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def beta_adder(init_beta, step_size=0.0001):
    beta = init_beta
    step_size = step_size

    def adder():
        nonlocal beta, step_size
        beta = min(beta + step_size, 1)
        return beta

    return adder
