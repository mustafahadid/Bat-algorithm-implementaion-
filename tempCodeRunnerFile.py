
#%%
from Functions import Ackley
from BAalgorithm import Bat_Algorithm



fn = Ackley()
optimiser = Bat_Algorithm(fn, Population_Size=100, Num_Movements=200)


%time _ = optimiser.Run(fn)

# display the results from the simulation

print('-'* 40)
print("Optimizer")

print('-'* 40)
print('Best Fitness: %.5f' %(optimiser.Best_Fitness))
print('Best Position: x: %.05f, y: %.5f' % (optimiser.Best_Position[0,0], optimiser.best_position[0, 1]))

print('-'* 40)
print("Function")
print('-'* 40)
print('Global Minimum: %.5f' % (fn.minima))
print('Location x: %.05f, y: %.5f' %(fn.location[0, 0], fn.location[0, 1]))
print('-'*40)

