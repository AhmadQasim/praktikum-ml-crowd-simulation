import tkinter as tk
import os
import subprocess


root = tk.Tk()
root.title("Exercise 1")
frame = tk.Frame(root)
frame.pack()


def task_2(label_out):
    label_out.config(text='Running Task 1...')
    root.update_idletasks()
    os.system('python3 tasks.py --grid_size_x=50 --grid_size_y=50 --timesteps=200 --mode="dijkstra" --p_num=5 '
              '--p_locs_mode="custom" --p_locs="[[5, 25]]" --t_locs="[[25, 25]]" --o_locs=[] '
              '--disable_pedestrians=1 simulate')
    label_out.config(text='Done')


def task_3(label_out):
    label_out.config(text='Running Task 3...')
    root.update_idletasks()
    os.system('python3 tasks.py --grid_size_x=50 --grid_size_y=50 --timesteps=200 --mode="dijkstra" --p_num=5 '
              '--p_locs_mode="circle" --p_locs="[[5, 25]]" --t_locs="[[25, 25]]" --o_locs=[] --p_locs_radius=20 '
              '--disable_pedestrians=1 simulate')
    label_out.config(text='Done')


def chicken_test(label_out):
    label_out.config(text='Running Chicken Test...')
    root.update_idletasks()
    os.system('python3 tasks.py --grid_size_x=50 --grid_size_y=50 --o_locs="[[14, 22], [15, 22], '
              '[15, 23], [15, 24], [15, 25], [15, 26], [15, 27], [15, 28], [14, 28]]" --timesteps=35 '
              '--p_locs_mode="custom" --p_locs_radius=20 --p_num=5 --p_locs="[[5, 23], [5, 25], [5, 27]]" '
              '--t_locs="[[25, 25]]" --mode="dijkstra" --disable_pedestrians=1 simulate')
    label_out.config(text='Done')


def bottle_neck(label_out):
    label_out.config(text='Running Bottleneck...')
    root.update_idletasks()
    os.system('python3 tasks.py --grid_size_x=50 --grid_size_y=50 --o_locs="[[9, 20], [8, 20], [7, 20], '
              '[6, 20], [5, 20], [4, 20], [3, 20], [3, 21], [3, 22], [3, 23], [3, 24], [15, 20], [14, 20], [13, 20], '
              '[12, 20], [11, 20], [10, 20], [15, 21], [15, 22], [15, 23], [17, 24], [16, 24], [15, 24], [15, 26], '
              '[17, 26], [16, 26], [15, 27], [15, 28], [15, 29], [15, 30], [14, 30], [13, 30], [12, 30], [11, 30], '
              '[10, 30], [3, 30], [3, 29], [3, 28], [3, 27], [3, 26], [3, 25], [3, 30], [9, 30], [8, 30], [7, 30], '
              '[6, 30], [5, 30], [4, 30]]" --timesteps=35 --p_locs_mode="custom" --p_locs_radius=20 --p_num=5 '
              '--p_locs="[[5, 23], [5, 25], [5, 27]]" --t_locs="[[25, 25]]" --mode="dijkstra" --disable_pedestrians=1 '
              'simulate')
    label_out.config(text='Done')


def task_5_test_1(label_out):
    label_out.config(text='Running Task 5-Test 1...')
    root.update_idletasks()
    os.system('python3 tasks.py --o_locs="[]" --timesteps=150 --p_locs_mode="custom" --p_locs_radius=20'
              '--p_num=5 --p_locs="[[2, 10]]" --t_locs="[[2, 110]]" --mode="normal" --grid_size_x=5 '
              '--grid_size_y=120 --disable_pedestrians=1 simulate')
    label_out.config(text='Done')


def task_5_test_2(label_out):
    label_out.config(text='Running Task 5-Test 2...')
    root.update_idletasks()
    result = subprocess.check_output('python3 task_5_test_5.py --c_locs="[[12, 1125], [12, 1250]]" --p_density=8 '
                                     '--o_locs="[]" --timesteps=200 --p_locs_mode="random" --p_locs_radius=20 '
                                     '--p_num=5 --p_locs="[[2, 10]]" --t_locs="[[12, 2499]]" --mode="dijkstra" '
                                     '--grid_size_x=25 --grid_size_y=2500 simulate', shell=True)
    label_out.config(text=result)


def task_5_test_3(label_out):
    label_out.config(text='Running Task 5-Test 3...')
    root.update_idletasks()
    os.system('python3 task_5_test_3.py --o_locs="[]" --timesteps=200 --p_locs_mode="uniform" '
              '--p_locs_radius=20 --p_num=20 --p_locs="[[2, 10]]" --t_locs="[[11, 38]]" --mode="dijkstra" '
              '--grid_size_x=50 --grid_size_y=50 --p_coord="[35, 10]" --p_region_x=5 --p_region_y=15 simulate')
    label_out.config(text='Done')


button = tk.Button(frame,
                   text="QUIT",
                   fg="red",
                   command=quit)
button.pack(side=tk.BOTTOM)

label = tk.Label(root, fg="dark green")
label.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="TASK 5-TEST 3",
                   command=lambda: task_5_test_3(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="TASK 5-TEST 2",
                   command=lambda: task_5_test_2(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="TASK 5-TEST 1",
                   command=lambda: task_5_test_1(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="BOTTLENECK",
                   command=lambda: bottle_neck(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="CHICKEN TEST",
                   command=lambda: chicken_test(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="TASK 3",
                   command=lambda: task_3(label))
slogan.pack(side=tk.BOTTOM)

slogan = tk.Button(frame,
                   text="TASK 2",
                   command=lambda: task_2(label))
slogan.pack(side=tk.BOTTOM)

root.mainloop()
