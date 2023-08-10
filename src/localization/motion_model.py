#!/usr/bin/env python

import numpy as np
import rospy


class MotionModel:

    def __init__(self):

        ####################################
        # Do any precomputation for the motion
        # model here.
        self.deterministic = rospy.get_param("~deterministic")
        self.num_particles = rospy.get_param("~num_particles")        
        #initialize variance
        self.x_v = 0.1 
        self.y_v = 0.1
        self.theta_v = 0.05

        #initialize transformation matrix
        self.d_matrix = np.zeros((self.num_particles,3))

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2]
        self.d_matrix[:,2] = dtheta

        particle_cos = np.cos(particles[:,2])
        particle_sin = np.sin(particles[:,2])

        '''from Q1A:

            [dxcos(dtheta) - dysin(dtheta)]
            [dxsin(dtheta) + dycos(dtheta)]
            [           dtheta            ]'''

        self.d_matrix[:,0] = dx*particle_cos - dy*particle_sin
        self.d_matrix[:,1] = dx*particle_sin + dy*particle_cos
        
        particles += self.d_matrix

        # add noise
        if self.deterministic == False:
            particles[:,0] += np.random.normal(0, self.x_v, self.num_particles)
            particles[:,1] += np.random.normal(0, self.y_v, self.num_particles)
            particles[:,2] += np.random.normal(0, self.theta_v, self.num_particles)

        return(particles)
