#!/usr/bin/env python

import numpy as np


class MotionModel:

    def __init__(self):

        ####################################
        # Do any precomputation for the motion
        # model here.

        self.deterministic = rospy.get_param("localization/deterministic")
        self.num_particles = rospy.get_param("localization/num_particles")
        
        #initialize variance
        self.v_pos = 0.5
        
        self.v_ang = 0.5

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
        self.d_matrix[:,1] = dx*particle_sin - dy*particle_cos
        
        particles += d_matrix

        if self.deterministic == False:
            #add noise
            particles[:,0] += np.random.normal(0, self.v_pos, self.num_particles)
            particles[:,1] += np.random.normal(0, self.v_pos, self.num_particles)
            particles[:,2] += np.random.normal(0, self.v_ang, self.num_particles)


        return(particles)
