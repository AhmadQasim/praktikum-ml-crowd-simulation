from utils import edit_scenario, plot_phase_portrait_first_part, create_plot, run_simulation
import os

if __name__ == '__main__':
    for i in range(1, 20, 1):
        plot_phase_portrait_first_part(163-68, filter(lambda f: not f.startswith('.') and not f.startswith('s'), os.listdir('../outputs/')),
                                   pedestrian_id=i)
    #create_plot(3)
    #y = 4.5
    #edit_scenario("Bottleneck bifurcation.scenario", y=y)
    #run_simulation("'/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/'",
     #              "../scenarios/'Bottleneck bifurcation.scenario'",
      #             f'../outputs/{y}/')
