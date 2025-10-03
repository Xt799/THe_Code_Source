#!/usr/bin/env python3

import os
import math
import time
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose2D, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Helper: angular difference (like MATLAB angdiff)

def angdiff(a, b):
    d = a - b
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return d


# exampleHelperComputeAngularVelocity

def exampleHelperComputeAngularVelocity(steeringDir, wMax=float('inf')):
    """
    Compute angular velocity (rough analogue of MATLAB helper).
    steeringDir: radians
    wMax: maximum allowed angular velocity
    """

    if wMax <= 0:
        raise ValueError("wMax must be positive")

    curPose_yaw = 0.0
    lookaheadPoint = (math.cos(steeringDir), math.sin(steeringDir))
    slope = math.atan2(lookaheadPoint[1] - 0.0, lookaheadPoint[0] - 0.0)
    alpha = angdiff(curPose_yaw, slope)

    w = 2.0 * math.sin(alpha)

    # If nearly opposite direction, pick a constant rotation
    if abs(abs(alpha) - math.pi) < 1e-12:
        w = math.copysign(1.0, w)

    if abs(w) > wMax:
        w = math.copysign(wMax, w)

    return w


# PIDgain 

def PIDgain(text):
    if text == "PID":
        kP = 1.2
        kI = 0.02
        kD = 0.01
    elif text == "P":
        kP = 1.2
        kI = 0.0
        kD = 0.0
    else:
        kP = 1.0
        kI = 0.0
        kD = 0.0
    return kP, kI, kD


def getCurrentPose2(topic):
    # Wait for a single Pose2D message from the topic
    msg = rospy.wait_for_message(topic, Pose2D, timeout=10.0)
    return np.array([msg.x, msg.y, msg.theta])



# setControllerP2P
# Returns (controller_callable, vfh_callable)

def setControllerP2P(goal):
    """
    goal: [x, y] as a 2-element iterable
    Returns:
      controller(robotCurrentPose) -> (v, omega, lookaheadPoint)
      vfh(scan, targetDir) -> steerDir (0 means ok)
    """

    # Controller parameters x   
    desiredLinearVelocity = 0.1
    MaxAngularVelocity = 1.0
    LookaheadDistance = 1.2

    def controller(robotCurrentPose):
        rx, ry, yaw = robotCurrentPose[0], robotCurrentPose[1], robotCurrentPose[2]
        gx, gy = goal[0], goal[1]
        
        # Distance to goal
        distance_to_goal = math.hypot(gx - rx, gy - ry)
        
        # Angle to goal
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        alpha = angdiff(angle_to_goal, yaw)
        
        # Always move forward if goal is reasonably aligned
        if abs(alpha) < math.pi / 1.5:
            v = desiredLinearVelocity
        else:
            # Goal is behind, move slower or rotate first
            v = desiredLinearVelocity * 0.3
        
        # Smooth angular control
        kp_ang = 1.5
        omega = max(-MaxAngularVelocity, min(MaxAngularVelocity, kp_ang * alpha))
        
        # Reduce speed when close to goal
        if distance_to_goal < 0.5:
            v *= (distance_to_goal / 0.5)
        
        lookaheadPoint = [gx, gy]
        return v, omega, lookaheadPoint



    # simple check for obstacles in front sectors
    class SimpleVFH:
        def __init__(self):
            self.UseLidarScan = True
            self.DistanceLimits = (0.10, 0.5)
            self.RobotRadius = 0.2
            self.MinTurningRadius = 0.2
            self.SafetyDistance = 0.4
            self.NumAngularSectors = 180
            self.HistogramThresholds = (3, 8)

        def __call__(self, scan, targetDir):
            """
            scan: sensor_msgs/LaserScan-like object with .ranges, .angle_min, .angle_max, .angle_increment
            targetDir: desired heading (radians), for simplified use it's 0 (robot frame)
            return: 0 if OK, else suggested steering direction angle (radians)
            """
            ranges = np.array(scan.ranges, dtype=float)
            # ignore NaNs and infs
            ranges = np.where(np.isfinite(ranges), ranges, np.inf)
            # convert angles
            angle_min = scan.angle_min
            angle_increment = scan.angle_increment
            angles = angle_min + np.arange(len(ranges)) * angle_increment
            # look at -30..+30 degrees for obstacles
            mask = np.abs(angles) <= math.radians(30)
            frontal = ranges[mask]
            if frontal.size == 0:
                return 0
            min_dist = np.min(frontal)
            if min_dist < self.SafetyDistance:
                # blocked: suggest turning direction away from nearest obstacle
                idx = np.argmin(frontal)
                angle_of_min = angles[mask][idx]
                # steer opposite direction (simple heuristic)
                suggested = -math.copysign(math.pi/4, angle_of_min)
                return suggested
            return 0

    vfh = SimpleVFH()
    return controller, vfh


def turnOrientation2(poseTopic, robotCmdPub, velMsg, dAngle):
    kP, kI, kD = PIDgain("P")
    Derivator, Integrator = 0.0, 0.0
    Integrator_max, Integrator_min = 10.0, -10.0

    pose = getCurrentPose2(poseTopic)
    Err = dAngle - pose[2]

    rate = rospy.Rate(10)
    while abs(Err) > 0.002 and not rospy.is_shutdown():
        pose = getCurrentPose2(poseTopic)
        Err = dAngle - pose[2]
        if Err > math.pi:
            Err -= 2*math.pi
        elif Err < -math.pi:
            Err += 2*math.pi

        P_value = kP * Err
        D_value = kD * (Err - Derivator)
        Derivator = Err
        Integrator = max(min(Integrator + Err, Integrator_max), Integrator_min)
        I_value = Integrator * kI
        PID = max(min(P_value + D_value + I_value, 0.8), -0.8)

        if abs(PID) < 0.01:
            break

        velMsg.linear.x = 0.0
        velMsg.angular.z = PID
        robotCmdPub.publish(velMsg)
        rate.sleep()

    velMsg.linear.x = 0.0
    velMsg.angular.z = 0.0
    robotCmdPub.publish(velMsg)
    rospy.sleep(0.1)

# movePoint2 

def movePoint2(lidarTopic, odomTopic, robotCmdPub, velMsg, goal, goalRadius):
    controller, vfh = setControllerP2P(goal)
    pose = getCurrentPose2(odomTopic)
    path = [goal[0], goal[1]]
    robotInitialLocation = [pose[0], pose[1]]
    robotGoal = path
    initialOrientation = pose[2]
    robotCurrentPose = np.array([robotInitialLocation[0], robotInitialLocation[1], initialOrientation])
    targetDir = 0.0
    distanceToGoal = np.linalg.norm(np.array(robotInitialLocation) - np.array(robotGoal))
    sampleTime = 0.1
    rate = rospy.Rate(1.0 / sampleTime)

    while distanceToGoal > goalRadius and not rospy.is_shutdown():
        robotCurrentPose = getCurrentPose2(odomTopic)
        v, omega, lookaheadPoint = controller(robotCurrentPose)
        laserScan = rospy.wait_for_message(lidarTopic, LaserScan)
        steerDir = vfh(laserScan, targetDir)

        if steerDir == 0:
            dV = v
            dW = omega
        else:
            # dV = 0.0
            # dW = exampleHelperComputeAngularVelocity(steerDir, 1.0)
            dV = v * 0.5  # Reduce speed but don't stop
            dW = omega + exampleHelperComputeAngularVelocity(steerDir, 0.5) 
        velMsg.linear.x = dV
        velMsg.angular.z = dW
        robotCmdPub.publish(velMsg)

        robotCurrentPose = getCurrentPose2(odomTopic)
        distanceToGoal = math.hypot(robotCurrentPose[0] - robotGoal[0], robotCurrentPose[1] - robotGoal[1])
        rate.sleep()

    velMsg.linear.x = 0.0
    velMsg.angular.z = 0.0
    robotCmdPub.publish(velMsg)


# Main script: toyfollowingros1
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard", data)
def toyfollowingros1():


    rospy.init_node('toy_following_node', anonymous=True)

    # Publishers / Subscribers
    robotCmdPub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    lidarTopic = '/scan'
    # mapPosTopic = '/map_pose'    
    # odomTopic = '/odom'      
    toyPoseTopic = '/toy_pose'   # geometry_msgs/Pose2D
    robotPoseTopic = '/robot_position' #

    # create a Twist message to reuse
    velMsg = Twist()


    # rospy.loginfo("Waiting for initial toy pose...")
    toyPoseMsg = None

    rospy.sleep(0.5)

    try:
        while not rospy.is_shutdown():

            q_start = getCurrentPose2(robotPoseTopic) 

            toyPoseMsg = rospy.wait_for_message(toyPoseTopic, Pose2D)
            q_goal = [toyPoseMsg.x, toyPoseMsg.y, 0.0]
            goalRadius = 0.4

            movePoint2(lidarTopic,robotPoseTopic,robotCmdPub,velMsg,q_goal[0:2],goalRadius)


            q_final = getCurrentPose2(robotPoseTopic)
            sweep_angle = math.atan2(q_goal[1] - q_final[1], q_goal[0] - q_final[0])

            turnOrientation2(robotPoseTopic, robotCmdPub, velMsg, sweep_angle)

            velMsg.linear.x = 0.0
            velMsg.angular.z = 1.0
            robotCmdPub.publish(velMsg)
            rospy.sleep(0.2)

            velMsg.linear.x = 0.0
            velMsg.angular.z = -1.0
            robotCmdPub.publish(velMsg)
            rospy.sleep(0.4)

            velMsg.linear.x = 0.0
            velMsg.angular.z = 1.0
            robotCmdPub.publish(velMsg)
            rospy.sleep(0.2)

            velMsg.linear.x = 0.0
            velMsg.angular.z = 0.0
            robotCmdPub.publish(velMsg)

    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure stop
        velMsg.linear.x = -0.0
        velMsg.angular.z = 0.0
        robotCmdPub.publish(velMsg)
        rospy.loginfo("toy_following_node stopped.")


if __name__ == "__main__":
    toyfollowingros1()
