
#%%
import numpy as np
from random import random, uniform
from numba import float32, int32
from numba.experimental import jitclass
from Functions import *

bat_algorithm_spec = [
    ('x_range', float32[:]),
    ('y_range', float32[:]),
    ('Population_Size', int32),
    ('Population', float32[:,:]),
    ('Fitness', float32[:,:]),
    ('Loudness', float32[:,:]),
    ('Pulse_rate', float32[:,:]),
    ('Velocity', float32[:,:]),
    ('Frequency', float32[:,:]),
    ('Movements', int32),
    ('Frequency_Range', float32[:]),
    ('Loudness_Decay', float32),
    ('Loudness_Limit', float32),
    ('Pulse_rate_Decay', float32),
    ('Gamma', float32),
    ('Best_Position', float32[:,:]),
    ('Best_Fitness', float32),
    ('Best_Bat', int32),
]


@jitclass(bat_algorithm_spec)

class Bat_Algorithm(object):
    def __init__(self, optimiser, Population_Size= 100, Num_Movements = 100):
        #function information
        self.x_range =optimiser.x_range
        self.y_range = optimiser.y_range

        # bat population and dynamics
        self.Population_Size = Population_Size
        self.Population, self.Fitness = self.Initialize_Population(optimiser) 
        
        # loudness, pulse rate, veloctiy,  frequency
        self.Loudness = np.ones(shape=(self.Population_Size, 1), dtype= np.float32)
        self.Pulse_rate = np.zeros(shape =(self.Population_Size, 1), dtype= np.float32)
        self.Frequency = np.zeros(shape =(self.Population_Size, 2), dtype= np.float32)
        self.Velocity = np.zeros(shape =(self.Population_Size, 2), dtype= np.float32)

        # create interal parameters
        self.Movements = Num_Movements
        self.Frequency_Range = np.array([0, 1], dtype=np.float32)
        self.Loudness_Decay = 0.5
        self.Loudness_Limit = 0.05
        self.Pulse_rate_Decay = 0.5
        self.Gamma = 0.5

        # optimication result fors
        self.Best_Position =np.empty(shape =(1, 2), dtype=np.float32)
        self.Best_Fitness = 0.0
        self.Best_Bat = 0

    def Initialize_Population(self, optimiser):
        Population = np.empty(shape =(self.Population_Size, 2), dtype=np.float32)
        Fitness = np.empty(shape =(self.Population_Size, 1), dtype=np.float32)

        for i in range(self.Population_Size):
            Population[i, 0] = (self.x_range[1] - self.x_range[0])* float32(random()) + self.x_range[0]
            Population[i, 1] = (self.y_range[1]- self.y_range[0])* float32(random()) + self.y_range[0]
            Fitness[i, 0] = optimiser.Query(Population[i, 0], Population[i, 1])

            return Population, Fitness

    # update the position method
    def Update_Dynamics(self, bat, best):
        self.Frequency[bat, :] = self.Frequency_Range[0] +(self.Frequency_Range[1] - self.Frequency_Range[0] * np.random.rand(1, self.Population.shape[1]))
        self.Velocity[bat, :] = (self.Population[best,:] - self.Population[bat,:])* self.Frequency[bat, :]
        Position = self.Population[bat, :] + self.Velocity[bat, :]
        return Position

    def Run(self, optimiser):
        best_fit = self.Fitness.min()
        best_index =self.Fitness.argmin()

        #Iterate through the simulation 
        for step in range (1, self.Movements + 1):
            for bat in range(self.Population_Size):
               
                #update the position of the bat
                Position = self.Update_Dynamics(bat, best_index)

                #Apply a random wald
                if random() < self.Loudness[bat, :]:
                    Position += self.Loudness.mean() * float32(uniform(-1, 1))
                
                #check limits of the search space
                if Position[0] <optimiser.x_range[0]:
                    Position[0] = optimiser.x_range[0]
                elif Position[0] >optimiser.x_range[1]:
                    Position[0] = optimiser.x_range[1]

                if Position[1] < optimiser.y_range[0]:
                    Position[1] = optimiser.y_range[0]
                elif Position[1] > optimiser.y_range[1]:
                    Position[1]= optimiser.y_range[1]
                
                # check the fitness of the bat
                Bat_Fitness = optimiser.Query(Position[0], Position[1])

                if Bat_Fitness < self.Fitness[bat, :]:
                    self.Fitness[bat,:] = Bat_Fitness
                    self.Population[bat,:] = Position
                    self.Pulse_rate[bat,:] = self.Pulse_rate_Decay*(1-np.exp(-self.Gamma *step))

                    if self.Loudness[bat, 0] > self.Loudness_Limit:
                        self.Loudness[bat, 0] *= self.Loudness_Decay
                    else:
                        self.Loudness[bat, 0] = self.Loudness_Limit
                    
                    if Bat_Fitness < best_fit:
                        best_fit = Bat_Fitness
                        best_index = bat

                        #update best values
                        self.Best_Position = Position.reshape(1, 2)
                        self.Best_Fitness = Bat_Fitness
                        self.Best_Bat = bat



