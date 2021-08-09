import shapely.geometry as sg
import shapely.affinity as sa
import numpy as np

import time
import copy

global sample
sample = False
arena = sg.box(-10,-10,10,10)
arena = arena.difference(sg.box(0,-1,11,1))
hole = sg.box(-7,-7,-5,7)
hole = hole.union(sg.box(-7,5,5,7))
hole = hole.union(sg.box(-7,-5,5,-7))
arena = arena.difference(hole)

walls = arena.boundary
iters = 0

global world
world = np.zeros(64)
goal = np.zeros(8192)
tau = 2*np.pi

def rotate_origin(obj, theta):
    return sa.rotate(obj,theta,use_radians=True,origin=(0,0))

bodySize = 0.3

#length of the sensors
sensors = [
    rotate_origin(sg.LineString([(0,bodySize),(0,1.5)]),-0.1*tau),
    rotate_origin(sg.LineString([(0,bodySize),(0,1.5)]),0.1*tau),
]

# for plotting only
bodyPoly = sg.Point(0,0).buffer(bodySize, resolution=2)
bodyPoly = bodyPoly.union(sg.box(-0.01,-1,0.01,0))

class AgentBody:
    # note: this class should only store what needs to be copied - everything
    # else (geometry etc.) should be generated on the fly
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.pos = sg.Point(x, y)
        self.theta = theta
    def randomValidPosition(self):
        while True:
            x, y = np.random.random(2)*24-12
            self.pos = sg.Point(x, y)
            self.theta =  np.random.random()*tau
            if self.inArena(): #and sum(self.sensorValues(False))==0:
                return self
    def inArena(self):
        return arena.contains(self.pos) and self.pos.distance(walls)>bodySize
    def sensorValues(self,plot=False,plot_sensors=False):
        mySensors = [
            sa.translate(rotate_origin(s,self.theta),self.pos.x,self.pos.y)
            for s in sensors
        ]
        result = np.zeros(len(mySensors))
        for i in range(len(mySensors)):
            if walls.intersects(mySensors[i]):
                result[i] = 1
        if plot_sensors:
            for i in range(len(mySensors)):
                # if result[i]:
                # 	col = '#aaffaa'
                # else:
                # 	col = '#cccccc'
                if result[i]:
                    col = '#00ff00'
                else:
                    col = '#000000'
                plot_line(ax,mySensors[i],col)
        if plot:
            if not self.inArena():
                col = '#ff0000'
            else:
                col = '#0000ff'
            body = sa.translate(rotate_origin(bodyPoly,self.theta),self.pos.x,self.pos.y)
            plot_poly(body,col)
        return result
    def update(self, controllerValues, dt=1.0):
        turnLeft, turnRight = controllerValues
        speed = (np.sum(controllerValues)+1)*0.2
        turning = 0.03*tau*(turnRight-turnLeft)
        self.theta += turning
        #print(self.pos, speed, controllerValues)
        self.pos = sa.translate(self.pos,
            -speed*np.sin(self.theta)*dt,speed*np.cos(self.theta))
        #print(self.pos)

class RNNController:
    # an FFController but with a feedback loop
    def __init__(self, n_inputs=3, n_outputs=2, n_hidden=5):
        # only store what needs to be copied
        self.network = FFController(n_inputs+n_hidden, n_outputs+n_hidden)
        self.state = np.zeros(n_hidden)
    def update(self,inputValues):
        # perform random movements for sampling the environment
        a,b = np.random.randint(0,2), np.random.randint(0,2)
        return a,b

class FFController:
    # feedforward controller
    def __init__(self, n_inputs=2, n_outputs=1):
        # only store what needs to be copied
        # (in this case weights only)
        self.W = np.zeros((n_outputs,n_inputs))
        self.b = np.zeros(n_outputs)
    def update(self,inputValues):
        return 0.5

class Agent:
    # note: this class should only store what needs to be copied - everything
    # else should be generated on the fly
    def __init__(self,it, lastValues = -1, lastlastValues = 0, x=0.0, y=0.0, theta=0.0):
        self.body = AgentBody(x,y,theta)
        self.controller = RNNController(len(sensors))
        self.it = it
        self.lastlastValues = lastlastValues
        self.lastValues = lastValues
    def alive(self):
        return self.body.inArena()
    def set(self, it):
        self.it = it
        self.lastValues = -1
        self.lastlastValues = 0
    def reset(self, it):
        self.body.randomValidPosition()
        self.it = it
        self.lastValues = -1
        self.lastlastValues = 0
        return self
    def update(self,dt=1.0,plot=False,plot_sensors=False):
        sensorValues = self.body.sensorValues(plot=plot,plot_sensors=plot_sensors)
        controllerValues = self.controller.update(sensorValues)
        # sample the world
        if(sample):
            if self.lastValues!= -1:
                wvalue = 0
                for i in range(2):
                    wvalue = wvalue + sensorValues[1 - i] * (2 ** i)
                wvalue = wvalue + (int(self.lastValues)* 4)
                wvalue = int(wvalue)
                world[ wvalue] = world[ wvalue] + 1

        # these are the values for sampling the goal
            self.lastValues =0
            for i in range(2):
                self.lastValues = self.lastValues + sensorValues[1-i]*(2**(2+i))
            self.lastValues = self.lastValues + controllerValues[1]*(2**0)
            self.lastValues = self.lastValues + controllerValues[0]*(2 ** 1)
            for i in range(2):
                self.lastlastValues = self.lastlastValues + sensorValues[1-i]*(2**(2+i))*(2**(8 - (self.it * 4)))
            self.lastlastValues = self.lastlastValues + controllerValues[1]*(2**0)*(2**(8 - (self.it * 4)))
            self.lastlastValues = self.lastlastValues + controllerValues[0] * (2 ** 1) * (2 ** (8 - (self.it * 4)))
        alive = self.alive()
        if alive:
            self.body.update(controllerValues)
        return alive



def main():
    plt.ion()
    fig.show()
    fig.canvas.draw()
    global world
    print(world)
    path = '/home/carlotta/Documents/RacecarCodeMultisensoryIntegration/inference05neu.py'
    output = open(path , 'w')
    output.write("import numpy as np")
    output.write('\n')
    n_agents = 15
    updates_per_generation = 3

    agents = [Agent(0).reset(0) for i in range(n_agents)]
    plt.cla()
    plot_arena()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(2)
    global sample
    sample = True
    for j in range(10000):
        plt.cla()
        plot_arena()
        if j%1000 == 0 : print('j', j)
        for i in range(n_agents):
            agents[i].set(0.0)
        #performing two steps
        for k in range(updates_per_generation-1):
            alive = [agent.update(plot=False) for agent in agents]
            for i in range(n_agents):
                agents[i].it = agents[i].it +1
        #third step to see whether agent is alive
        for i in range(n_agents):
            alive[i] = agents[i].update(plot=True,plot_sensors=True)
            agents[i].it = agents[i].it + 1

        print(alive)
        # safe the results in the goal array
        for i in range(n_agents):
            val = agents[i].lastlastValues + alive[i]*2**12
            goal[val.astype(int)] = goal[val.astype(int)] + 1

        [agent.reset(0) for agent in agents]
        alive = [True for i in range(n_agents)]
        fig.canvas.draw()
        fig.canvas.flush_events()
    # write results to the file in the "path"
    my_string = ','.join(map(str, goal))
    world = ','.join(map(str, world))
    output.write("goal = np.array([")
    output.write(my_string)
    output.write(']) \n')
    output.write("world = np.array([")
    output.write(world)
    output.write("])")
    output.close()
    return



def plot_arena():
    plot_poly(arena)
    x_range = walls.bounds[0]-1,walls.bounds[2]+1
    y_range = walls.bounds[1]-1,walls.bounds[3]+1
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect(1)


import matplotlib
# the next line is needed on my machine to make animation work - if you have
# problems on your machine, try changing it to a different MatPlotLib backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(111)

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }
def v_color(ob):
    return COLOR[ob.is_valid]
def plot_poly(polygon, col=None):
    if not col:
        col = v_color(polygon)
#	patch = PolygonPatch(polygon, facecolor=col, edgecolor=col, alpha=1, zorder=2)
    patch = PolygonPatch(polygon, fill=False, edgecolor=col, alpha=1, zorder=2)
    ax.add_patch(patch)
def plot_line(ax, ob, col='#000000'):
    x, y = ob.xy
    ax.plot(x, y, color=col, alpha=1, linewidth=1, solid_capstyle='round', zorder=2)


main()

