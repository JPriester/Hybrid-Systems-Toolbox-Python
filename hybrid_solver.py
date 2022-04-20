"""
A python implementation of the Hybrid Systems Simulation Toolbox

@author: Jan de Priester
"""

import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['text.usetex'] = True

class Hybrid_solver:
    """A python implementation of the Hybrid Systems Simulation Toolbox
    
    Solves the evolution of the hybrid system over a horizon of 
    (t,j) ∈ (t_max, j_max). 
    The solver expects 4 functions:
        - The flow map F(t,z): A function that describes the continuous 
            evolution of the state z when in the flow set. The function has to 
            return a list containing the derivative w.r.t time of each 
            component of z.
        - The jump map G(z): A function that describes the discontinuous jumps 
            of the state z when in the jump set. The function has to return a 
            list containing the values to which z has to be reset to.
        - the flow set C(t,z): a function that return 1 when in the flow set
            and returns 0 when not in the flow set.
        - the jump set D(t,z): a function that returns 0 when in the jump set
            and returns 1 when not in the jump set. 
            IMPORTANT: the jump set definition is different than is
            used in the Matlab implementation. 
    Other parameters the solver expects:
        - z_0: a list containing the initial condition of the state z.
        - t_max: the horizon of ordinairy time t.
        - j_max: the maximum allowable jumps j.
        Side note: the solver stops when either t>t_max or j>j_max
        - rule: the priority rule. It can occurs that the system is both
            in the flow set and the jump set at the same hybrid time. In this
            case, one must define a priority to flow or jump, or set this to 
            be picked at random. Hence, rule accepts three arguments:
                - 'flow': give priority to flow (default).
                - 'jump': give priority to jump.
                - 'random': random priority.
        - plot2d: plot the hybrid arc in a 2 dimensional figure. Creates two
            figures in a subfigure environment. Plots the hybrid state z over 
            time t and plots the hybrid state z over jumps j.
        - plot3d: plot the hybrid arc in a 3 dimensional figure. Plots the 
            hybrid state z over time t and jumps j.
        - max_step: maximum step setting used for scipy.integrate.
        - rtol: relative tolerance setting used for scipy.integrate.    
    """
    
    def __init__(self, flow_map, jump_map, flow_set, jump_set, z_0, 
                 t_max, j_max, rule='flow', plot2d=True, plot3d=True, 
                 max_step=0.001, rtol=1e-8):
        self.flow_map = flow_map
        self.jump_map = jump_map
        self.flow_set = flow_set
        self.jump_set = jump_set
        self.t_max = t_max
        self.j_max = j_max
        self.rule = rule
        self.max_step = max_step
        self.rtol = rtol
        
        self.z = z_0
        self.flow_set.terminal = True
        self.jump_set.terminal = True
        self.stop = False
        self.ts = []
        self.js = []
        self.ys = []
        
        self.t = 0
        self.j = 0
        
        if self.rule == 'flow':
            self.events = self.flow_set
        elif self.rule == 'jump':
            self.events = self.jump_set
        elif self.rule == 'random':
            self.events = [self.flow_set, self.jump_set]
        else:
            print('Typo or no priority rule given!')
        
        self.solve_ode()
        self.fig_number = 1
        if plot2d:
            self.plot_hybrid_arc_2d()
        if plot3d:
            self.plot_hybrid_arc_3d()
        
        self.sol_t = np.concatenate(self.ts)
        self.sol_j = np.concatenate(self.js)
        self.sol_z = np.concatenate(self.ys, axis=1)
        
    def solve_ode(self):
        while True:              
            sol = integrate.solve_ivp(self.flow_map, (self.t, self.t_max), 
                                      self.z, events=self.events,
                                      max_step = self.max_step, 
                                      rtol=self.rtol)
            self.ts.append(sol.t)
            self.ys.append(sol.y)
            self.js.append([self.j]*len(sol.t))
            if sol.status == 1: # Event was hit
                # New start time for integration
                self.t = sol.t[-1]
                # Reset initial state
                self.z = sol.y[:, -1].copy()
                
                if self.rule=='flow':
                    self.priority_flow()
                elif self.rule=='jump':
                    self.priority_jump()
                elif self.rule =='random':
                    rand_integer = np.random.randint(2)
                    if rand_integer == 0:
                        self.priority_flow()
                    else:
                        self.priority_jump()
                
                if self.stop:
                    break
                if self.j > self.j_max:
                    break
            else:
                break
    
    def priority_flow(self):
        # flow priority: if we can't flow, we jump
        if self.flow_set(1, self.z) == 0:
            if self.jump_set(1,self.z) == 0:
                self.z = self.jump_map(self.z)
                self.j += 1
            else:
                # we can't flow or jump
                print('Hybrid state is in neither the flowset or jumpset!')
                self.stop = True
            
    
    def priority_jump(self):
        # jump priority: if we can't jump, we check if we can flow
        if self.jump_set(1, self.z) == 0:
            self.z = self.jump_map(self.z)
            self.j += 1
        elif self.flow_set(1,self.z) == 0:
            # we can't flow or jump
            print('Hybrid state is in neither the flowset or jumpset!')
            self.stop = True
            
    def plot_hybrid_arc_2d(self):
        for state in range(len(self.ys[0])):
            fig = plt.figure(num = self.fig_number)
            self.fig_number += 1
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            for index in range(len(self.ys)):
                ax1.plot(self.ts[index],
                        self.ys[index][state,:], color='blue')
                ax2.plot(self.js[index], self.ys[index][state,:], color='blue'
                         , linestyle='--')
                if index < len(self.ys)-1:
                    ax1.plot([self.ts[index][-1], self.ts[index+1][0]],
                             [self.ys[index][state,-1], 
                              self.ys[index+1][state,0]], color='red', 
                             linestyle='--')
                    ax2.plot([self.js[index][-1], self.js[index+1][0]],
                              [self.ys[index][state,-1], 
                              self.ys[index+1][state,0]], color='red', 
                              marker='.', markersize=10, linestyle='none')
                else:
                    ax1.plot(self.ts[index][-1],
                        self.ys[index][state,-1], color='blue', marker='.',
                        markersize=10)   
                    # ax2.plot(self.js[index][-1],
                    #     self.ys[index][state,-1], color='red', marker='.',
                    #     markersize=10)  

            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax1.tick_params(axis='both', which='minor', labelsize=12)
            ax1.set_xlabel('flows $[t]$', fontsize=18)
            ax1.set_ylabel(f"$z_{state+1}$", fontsize=22)
            ax1.grid()
            
            ax2.tick_params(axis='both', which='major', labelsize=14)
            ax2.tick_params(axis='both', which='minor', labelsize=12)
            ax2.set_xlabel('jumps $[j]$', fontsize=18)
            ax2.set_ylabel(f"$z_{state+1}$", fontsize=22)
            ax2.grid()
            
            fig.tight_layout()
    
    def plot_hybrid_arc_3d(self):
        for state in range(len(self.ys[0])):
            fig = plt.figure(figsize=(8,5), num = self.fig_number)
            self.fig_number += 1
            fig.subplots_adjust(bottom=-0.01,top=1)
            ax = fig.add_subplot(111, projection='3d')
            for index in range(len(self.ys)):
                ax.plot(self.ts[index], self.js[index], 
                        self.ys[index][state,:], color='blue')
                if index < len(self.ys)-1:
                    ax.plot([self.ts[index][-1], self.ts[index+1][0]], 
                            [self.js[index][-1], self.js[index+1][0]],
                            [self.ys[index][state,-1], 
                             self.ys[index+1][state,0]], color='red',
                            linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.set_ylabel('$j$', fontsize=22)
            ax.set_xlabel('$t$', fontsize=22)
            ax.set_zlabel(f"$z_{state+1}$", fontsize=22)
            plt.gca().invert_xaxis()
            ax.view_init(25, 90+22)