from utils import edit_scenario, plot_phase_portrait_first_part, create_plot, run_simulation
import os

if __name__ == '__main__':
    plot_phase_portrait_first_part(163-68, filter(lambda f: not f.startswith('.'), os.listdir('./outputs/')),
                                   pedestrian_id=3)
    create_plot(3)
    y = 4.5
    edit_scenario("Bottleneck bifurcation.scenario", y=y)
    run_simulation("'/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/'",
                   "../scenarios/'Bottleneck bifurcation.scenario'",
                   f'../outputs/{y}/')
