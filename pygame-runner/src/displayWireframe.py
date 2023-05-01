# from https://www.petercollingridge.co.uk/tutorials/3d/pygame/matrix-transformations/

import wireframe as wf
import pygame
import numpy as np

key_to_function = {
    # pygame.K_LEFT:   (lambda x: x.translateAll('x', -10)),
    # pygame.K_RIGHT:  (lambda x: x.translateAll('x',  10)),
    # pygame.K_DOWN:   (lambda x: x.translateAll('y',  10)),
    # pygame.K_UP:     (lambda x: x.translateAll('y', -10)),
    pygame.K_LEFT: (lambda x: x.translateAll([-10, 0, 0])),
    pygame.K_RIGHT:(lambda x: x.translateAll([ 10, 0, 0])),
    pygame.K_DOWN: (lambda x: x.translateAll([0,  10, 0])),
    pygame.K_UP:   (lambda x: x.translateAll([0, -10, 0])),

    pygame.K_EQUALS: (lambda x: x.scaleAll(1.25)),
    pygame.K_MINUS:  (lambda x: x.scaleAll(0.80)),

    pygame.K_s:      (lambda x: x.rotateAll('X',  0.1)),
    pygame.K_w:      (lambda x: x.rotateAll('X', -0.1)),
    pygame.K_a:      (lambda x: x.rotateAll('Y',  0.1)),
    pygame.K_d:      (lambda x: x.rotateAll('Y', -0.1)),
    pygame.K_q:      (lambda x: x.rotateAll('Z',  0.1)),
    pygame.K_e:      (lambda x: x.rotateAll('Z', -0.1))}

class ProjectionViewer:
    """ Displays 3D objects on a Pygame screen """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Wireframe Display')
        self.background = (10,10,50)

        self.wireframes = {}
        self.displayNodes = True
        self.displayEdges = True
        self.nodeColour = (255,255,255)
        self.edgeColour = (200,200,200)
        self.nodeRadius = 2

    def addWireframe(self, name, wireframe):
        """ Add a named wireframe object. """

        self.wireframes[name] = wireframe

    def run(self):
        """ Create a pygame screen until it is closed. """

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in key_to_function:
                        key_to_function[event.key](self)
                    
            self.display()  
            pygame.display.flip()
        
    def display(self):
        """ Draw the wireframes on the screen. """

        #print("in display")
        self.screen.fill(self.background)

        for wireframe in self.wireframes.values():
            #print("in wireframe display loop")
            if self.displayEdges:
                for n1, n2 in wireframe.edges:
                    pygame.draw.aaline(self.screen, self.edgeColour, wireframe.nodes[n1][:2], wireframe.nodes[n2][:2], 1)

            if self.displayNodes:
                for node in wireframe.nodes:
                    pygame.draw.circle(self.screen, self.nodeColour, (int(node[0]), int(node[1])), self.nodeRadius, 0)

    def translateAll(self, vector):
        """ Translate all wireframes along a given axis by d units. """
        for wireframe in self.wireframes.values():
            matrix = wireframe.translationMatrix(*vector)
            wireframe.transform(matrix)

    def scaleAll(self, scale):
        for wireframe in self.wireframes.values():
            matrix = wireframe.scaleMatrix(scale)
            wireframe.transform(matrix)

    def rotateAll(self, axis, theta):
        rotateFunction = 'rotate' + axis
        for wireframe in self.wireframes.values():
            matrix = getattr(wireframe, rotateFunction)(theta)
            wireframe.transform(matrix)

if __name__ == '__main__':
    print("Starting....")
    pv = ProjectionViewer(800, 600)

    cube = wf.Wireframe()
    cube_nodes = [(x,y,z) for x in (50,250) for y in (50,250) for z in (50,250)]
    cube.addNodes(np.array(cube_nodes))
    cube.addEdges([(n,n+4) for n in range(0,4)]+[(n,n+1) for n in range(0,8,2)]+[(n,n+2) for n in (0,1,4,5)])
    
    pv.addWireframe('cube', cube)
    print("calling pv.run")
    pv.run()