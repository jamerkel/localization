#!/usr/bin/env python

import numpy as np
import time
from threading import RLock

import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, PoseStamped, PoseArray, Quaternion, PolygonStamped, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from std_msgs.msg import Float32
import tf
import tf.transformations
import tf2_ros as tf2
import tf2_geometry_msgs

from sensor_model import SensorModel
from motion_model import MotionModel
import utils as Utils
#from localization.msg import ConvergenceTracker

class ParticleFilter:
    '''
    Monte Carlo Localization based on odometry and a laser scanner.
    '''
    def __init__(self):
        self.num_particles = int(rospy.get_param("~num_particles"))
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")
        self.viz_particles = int(rospy.get_param("~viz_particles"))
        # self.PUBLISH_ODOM = bool(rospy.get_param("~publish_odom", "1"))
        self.angle_step = int(rospy.get_param("~angle_step"))
        self.visualizing = bool(rospy.get_param("~visualizing"))
        # tracking 
        self.lidar_init = False
        self.odom_init = False
        self.map_init = False
        self.last_odom_pose = None 
        self.last_odom_msg = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.lock = RLock()
        # models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()
        # particles
        self.inferred_pose = None
        self.particles = np.zeros((self.num_particles, 3))
        self.particle_indices = np.arange(self.num_particles)
        self.weights = np.ones(self.num_particles) / float(self.num_particles)
        self.particle_init()
        # map
        self.permissible_region = None
        # visualization
        self.inferred_pose_pub = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1)
        self.particle_pub = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1)
        self.odom_pub = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        # transform topics
        self.pub_tf = tf.TransformBroadcaster()
        self.tf_buffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tf_buffer)
        self.inferred_pose_sub = rospy.Subscriber("/pf/viz/inferred_pose", PoseStamped, self.inferred_pose_cb)
        # laser data
        self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.lidar_cb, queue_size=1)
        # self.odom_sub  = rospy.Subscriber(rospy.get_param("~odom_topic", "/vesc/odom"), Odometry, self.callback_odom, queue_size=1) # when using controller
        self.odom_sub  = rospy.Subscriber(rospy.get_param("~odom_topic", "/odom"), Odometry, self.odom_cb, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.pose_click_cb, queue_size=1)
        # algorithm analysis
        self.convergence_pub = rospy.Publisher("/convergence", Point, queue_size=10)
        # errors
        self.xy_error_pub = rospy.Publisher('/position_err', Float32, queue_size=1)
        self.theta_error_pub = rospy.Publisher('/ang_error', Float32, queue_size=1)
        self.x_gt_pub = rospy.Publisher('gt_x', Float32, queue_size=1)
        self.y_gt_pub = rospy.Publisher('gt_y', Float32, queue_size=1)
        self.theta_gt_pub = rospy.Publisher('gt_theta', Float32, queue_size=1)
        self.x_inferred_pub = rospy.Publisher('inferred_x', Float32, queue_size=1)
        self.y_inferred_pub = rospy.Publisher('inferred_y', Float32, queue_size=1)
        self.theta_inferred_pub = rospy.Publisher('inferred_theta', Float32, queue_size=1)

    def particle_init(self):
        '''
        Initialize particles.
        '''
        print('initializing particles')
        with self.lock:
            while not self.sensor_model.map_set:
                rospy.sleep(0.05)
            self.map_init = True
            self.permissible_region = self.sensor_model.permissible_region
            self.map = self.sensor_model.map
            self.map_info = self.sensor_model.map_info
            permissible_x, permissible_y = np.where(self.permissible_region == 1)
            num_permissible = len(permissible_x)
            # Initial set of particles 
            initial_particles = np.zeros((self.num_particles, 3))
            selected_indices = np.random.choice(num_permissible, size=self.num_particles)
            initial_particles[:, 0] = permissible_y[selected_indices]
            initial_particles[:, 1] = permissible_x[selected_indices]
            initial_particles[:, 2] = np.random.uniform(0, 2*np.pi, size=self.num_particles)
            # Transform particles from grid to world coords
            Utils.map_to_world(initial_particles, self.map_info)
            self.particles = initial_particles
            self.weights[:] = 1.0 / self.num_particles

    def pose_click_cb(self, msg):
        '''
        Initialize particle distribution from clicked pose.
        '''
        pose = msg.pose.pose
        with self.lock:
            self.weights = np.ones(self.num_particles) / float(self.num_particles)
            self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.num_particles)
            self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.num_particles)
            self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.num_particles)
        # Uncomment for simulation
        if isinstance(self.last_odom_pose, np.ndarray):
            self.last_odom_pose[0] = pose.position.x
            self.last_odom_pose[1] = pose.position.y
            self.last_odom_pose[2] = Utils.quaternion_to_angle(pose.orientation)

    def publish_tf_odom(self, pose, stamp=None):
        """ Publish a tf for the car. This tells ROS where the car is with respect to the map. """
        if stamp == None:
            stamp = rospy.Time.now()
        self.pub_tf.sendTransform((pose[0],pose[1],0),tf.transformations.quaternion_from_euler(0, 0, pose[2]),
               stamp , self.particle_filter_frame, "/map")
        # update odom
        odom = Odometry()
        odom.header = Utils.make_header("/map", stamp)
        odom.pose.pose.position.x = pose[0]
        odom.pose.pose.position.y = pose[1]
        odom.pose.pose.orientation = Utils.angle_to_quaternion(pose[2])
        self.odom_pub.publish(odom)

    def publish_particles(self, particles):
        pa = PoseArray()
        pa.header = Utils.make_header("map")
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)
        #compute particle spread
        self.convergence_tracking(pa)

    def visualize(self):
        '''
        Publish various visualization messages.
        '''
        if not self.visualizing:
            return
        if self.inferred_pose_pub.get_num_connections() > 0 and isinstance(self.inferred_pose, np.ndarray):
            ps = PoseStamped()
            ps.header = Utils.make_header("map")
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.inferred_pose_pub.publish(ps)
        if self.particle_pub.get_num_connections() > 0:
            if self.num_particles > self.viz_particles:
                # randomly downsample
                proposal_indices = np.random.choice(self.particle_indices, self.viz_particles, p=self.weights)
                self.publish_particles(self.particles[proposal_indices,:])
            else:
                self.publish_particles(self.particles)

    def lidar_cb(self, msg):
        '''
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        '''
        if not isinstance(self.laser_angles, np.ndarray):
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = np.copy(self.laser_angles[0::self.angle_step]).astype(np.float32)
        self.downsampled_ranges = np.array(msg.ranges[::self.angle_step])
        self.lidar_init = True

    def odom_cb(self, msg):
        '''
        Track changing odom data.
        '''
        if self.last_odom_msg:
            last_time = self.last_odom_msg.header.stamp.to_sec()
            this_time = msg.header.stamp.to_sec()
            dt = this_time - last_time
            self.odometry_data = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])*dt
            self.last_odom_msg = msg
            self.last_stamp = msg.header.stamp
            self.update()
        else:
            self.last_odom_msg = msg
            self.last_stamp = msg.header.stamp
            self.odom_init = True

    def inferred_pose_cb(self, msg):
        '''
        Calculate euclidean distance error and anglular error,
        and publish inferred pose components for direct comparison.
        '''
        try:
            # Get the ground truth position from the transformation
            transform = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0))

            inferred_x = self.inferred_pose[0]
            inferred_y = self.inferred_pose[1] 
            inferred_orientation = self.inferred_pose[2]

            ground_truth_x = transform.transform.translation.x
            ground_truth_y = transform.transform.translation.y
            ground_truth_orientation = tf.transformations.euler_from_quaternion([
                                            transform.transform.rotation.x,
                                            transform.transform.rotation.y,
                                            transform.transform.rotation.z,
                                            transform.transform.rotation.w])[2]
            # euclidean error
            euclidean_error = np.linalg.norm(np.array([ground_truth_x, ground_truth_y]) - np.array([inferred_x, inferred_y]))
            self.xy_error_pub.publish(euclidean_error)
            # angular error
            angular_error = self.angle_diff(ground_truth_orientation, inferred_orientation)
            self.theta_error_pub.publish(angular_error)
            # direct comparison
            self.x_gt_pub.publish(ground_truth_x)
            self.y_gt_pub.publish(ground_truth_y)
            self.theta_gt_pub.publish(ground_truth_orientation)
            self.x_inferred_pub.publish(inferred_x)
            self.y_inferred_pub.publish(inferred_y)
            self.theta_inferred_pub.publish(inferred_orientation)

            # print("Anglular Error: ", angular_error)
            # print("Euclidean Distance Error:", euclidean_error)

        except Exception as e: 
            print("\nSomething happened: ", e, "\n")
            pass

    def angle_diff(self, angle1, angle2):
        """
        Calculate the smallest difference between two angles in radians.
        """
        diff = angle1 - angle2
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return np.abs(diff)

    def convergence_tracking(self, particles):
        convergence_msg = Point()
        #convergence_msg.header = Utils.make_header("base_link")

        max_x = max([pose.position.x for pose in particles.poses])
        min_x = min([pose.position.x for pose in particles.poses])
        x_dist = max_x - min_x

        max_y = max([pose.position.y for pose in particles.poses])
        min_y = min([pose.position.y for pose in particles.poses])
        y_dist = max_y - min_y

        tot_dist = np.sqrt(x_dist**2 + y_dist**2)

        convergence_msg.x = x_dist
        convergence_msg.y = y_dist
        convergence_msg.z = tot_dist

        self.convergence_pub.publish(convergence_msg)

    def MCL(self, odom, scans):
        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        # get sample
        sample_indices = np.random.choice(self.particle_indices, self.num_particles, p=self.weights)
        sample = self.particles[sample_indices,:]
        # update particles sample with motion model
        self.particles = self.motion_model.evaluate(sample, odom)
        # update weights with sensor model and normalize
        self.weights = self.sensor_model.evaluate(self.particles, scans)
        self.weights /= np.sum(self.weights)
        # compute inferred pose
        self.inferred_pose = np.dot(self.particles.transpose(), self.weights)
        # publish transform
        self.publish_tf_odom(self.inferred_pose, self.last_stamp)

        return True  # on success 

    def update(self):
        '''
        Apply the MCL function to update particle filter state. 
        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if self.lidar_init and self.odom_init and self.map_init:
            with self.lock:
                # Run MCL with current scan and odom data
                scans = np.copy(self.downsampled_ranges).astype(np.float32)
                odom = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)
                self.MCL(odom,scans)
            # visualize on update 
            self.visualize()


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()