from utils import plot_saddle_bifurcation, edit_scenario, run_simulation, plot_phase_portrait_second_part
import numpy as np


if __name__ == '__main__':
    '''
    for d in np.arange(0.0, 5.0, 0.2):
        edit_scenario("./Saddle_Node_Bifurcation.scenario", d=d)
        run_simulation('"/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/"',
                       "../scenarios/Saddle_Node_Bifurcation.scenario", f'../outputs/saddle_d{d}/')
    '''

    # plot_saddle_bifurcation(np.around(np.arange(0.0, 5.0, 0.2), decimals=1))
    plot_phase_portrait_second_part(5, np.around(np.arange(0.0, 5.0, 0.2), decimals=1))
