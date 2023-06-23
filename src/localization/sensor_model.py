#!/usr/bin/env python

import numpy as np
from localization.scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.num_particles = rospy.get_param("~num_particles")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")


        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.zmax = self.num_particles
        self.table_width = self.zmax + 1

        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        length = self.table_width
        self.sensor_model_table = np.zeros((length, length))
        for d in range(length):
            #normalize p_hit, separately normalize p_tot
            p_hit_tot = 0.
            p_hit_arr = []
            #p_hit normalized later
            norm = self.alpha_hit

            for z in range(length):
                p = 0.
                p_hit = self.alpha_hit*np.exp(-float(z-d)**2/(2.*self.sigma_hit**2))/np.sqrt(2.0*np.pi)
                p_hit_tot += p_hit
                p_hit_arr.append(p_hit)
                
                #p_short
                if z <= d and d != 0:
                    p += self.alpha_short*(2./d)*(1-(z/d))
                #p_max
                if z == self.zmax:
                    p += self.alpha_max

                #p_rand
                p += self.alpha_rand/float(self.zmax)

                self.sensor_model_table[z,d] = prob
                norm += p
            #normalize all p_hit values for every value of z
            self.sensor_model_table[:, d] += p_hit_arr/p_hit_tot
            #normalize all p values for every value of z
            self.sensor_model_table[:, d] /= norm



    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scaling = float(self.map_resolution * self.lidar_scale_to_map_scale)
        scans = self.scan_sim.scan(particles) / scaling
        observation /= scaling

        observation[i>self.zmax] = self.zmax
        observation[i<0] = 0

        scans[i>self.zmax] = self.zmax
        scans[i<0] = 0

        scans_int = np.rint(scans)
        obs_int = np.rint(observation)

        probabilities = np.prod(self.sensor_model_table[scans_int, obs_int] axis=1)
        prob_squash = np.power(probabilities, 1./2.2)

        return(prob_squash)


        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
