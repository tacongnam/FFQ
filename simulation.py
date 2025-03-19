import sys

from simulator import parameters as para
from simulator.import_network import Simulation

def setup_parameters():
    para.energy_q = 0.8
    para.connect_q = 0.1
    para.cover_q = 0.1

def main():
    testcase_name = 'hanoi1000n50_allconnect'
    if len(sys.argv) == 2:
        testcase_name = sys.argv[1]

    sim = Simulation(f'data/{testcase_name}.yaml')
    sim.network_init(para.q_alpha, para.q_gamma)
    sim.run_simulator(run_times=1, E_mc=54000, num_test=1)

if __name__ == "__main__":
    main()