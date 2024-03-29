import argparse
import os
import time
import json

import numpy as np
import tensorflow as tf

from run_dynamical import model_fn, input_fn

import rps.robotarium as robotarium
import rps.utilities.graph as graph
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *


class ChaserAgents:

    def __init__(self,num_agents, max_vel, max_acc, max_ang_vel, radius, time_step, model_params, log_dir, show_figure = True, Kp = 1, Kd = 0, Ki = 0):

        self._next_positions = np.zeros((2,num_agents))
        self.velocities = np.zeros((2,num_agents))
        self.orientations = np.zeros((1,num_agents))

        self.poses_robotarium = np.zeros((3,num_agents))

        self.num_agents = num_agents
        self.time_step = time_step
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_ang_vel = max_ang_vel

        self.Kp = Kp
        self.Ki = Ki*time_step
        self.Kd = Kd/time_step

        for i in range(num_agents):

            alpha = i*2*np.pi/num_agents # angle from the center
            theta = alpha + np.pi/2.0 # heading angle

            if theta > 2*np.pi:
                theta = theta - 2 * np.pi

            if theta > np.pi:
                theta = theta - 2 *np.pi

            x = radius * np.cos(alpha)
            y = radius * np.sin(alpha)

            self.poses_robotarium[:,i] = np.array([x, y, theta]).transpose()

        self.r = robotarium.Robotarium(number_of_agents= num_agents, show_figure= show_figure, save_data= False, update_time = 1) #, update_time= time_step)

        self.go_to_point_and_pose(self.poses_robotarium)

        self.uni_barrier_cert = create_unicycle_barrier_certificate(num_agents, barrier_gain = 100, safety_radius= 0.1650, projection_distance = 0.05)

        #pos_vel_log
        current_position = self.r.get_poses()[:2,:]
        current_velocities = np.zeros((2,self.num_agents))
        self.r.set_velocities(np.arange(self.num_agents), current_velocities)
        self.r.step()
        current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)

        self.pos_vel_log = np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)

        #define GNN
        self.seg_len = 2 * len(model_params['cnn']['filters']) + 1
        self.cnn_multistep_regressor = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir= log_dir)




    def go_to_point_and_pose(self, goal_points):

        uni_barrier_cert = create_unicycle_barrier_certificate(self.num_agents, safety_radius=0.05)

        # define x initially
        x = self.r.get_poses()
        self.r.step()

        while(np.size(at_pose(x, goal_points)) != self.num_agents):

            # Get poses of agents
            x = self.r.get_poses()

            # Unicycle control inputs
            dxu = unicycle_pose_controller(x, goal_points)

            # Create safe input s
            dxu = uni_barrier_cert(dxu, x)

            self.r.set_velocities(np.arange(self.num_agents), dxu)
            self.r.step()


    def move_agents(self):

        self.poses_robotarium = self.r.get_poses()
        self.r.step()

        current_velocities = np.zeros((2,self.num_agents))

        for i in range(self.num_agents):

            j = i+1
            if i == self.num_agents -1:
                j = 0
            agent_i = self.poses_robotarium[:2,i]
            agent_j = self.poses_robotarium[:2,j]

            acc = (agent_j - agent_i) * 2.0 / self.time_step**2

            if np.linalg.norm(acc) > self.max_acc:
                acc *=  self.max_acc / np.linalg.norm(acc)

            vel = acc * self.time_step

            if np.linalg.norm(vel) > self.max_vel:
                vel *=  self.max_vel / np.linalg.norm(vel)

            self._next_positions[:,i] = self.poses_robotarium[:2,i] + self.time_step * vel

            cos_sin = vel/np.linalg.norm(vel)
            self.orientations[:,i] = np.arccos(cos_sin[:1]) * np.sign(cos_sin[1:])

            self.velocities[0,i] = np.linalg.norm(vel)
            current_velocities[:,i] = vel

        self.go_to_pose_and_control()

        #pos_vel_log
        current_position = self.r.get_poses()[:2,:]
        self.r.set_velocities(np.arange(self.num_agents), np.zeros((2,self.num_agents)))
        self.r.step()
        current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)
        self.pos_vel_log = np.concatenate([self.pos_vel_log,
                                np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)], axis = 1)


    def move_agents_GNN(self):

        current_step = self.pos_vel_log.shape[1]

        test_data = self.pos_vel_log[:,current_step-8:current_step,:,:]
        test_data = test_data.astype(dtype = np.float32)

        features = {'time_series': test_data}
        predict_input_fn = input_fn(features, self.seg_len, 1, 1, 'test')

        prediction = self.cnn_multistep_regressor.predict(input_fn = predict_input_fn)
        prediction = np.array([pred['next_steps'] for pred in prediction])

        goal_velocities = np.squeeze(np.swapaxes(prediction,2,3)[:,:,2:,:])
        goal_velocities = self.regularize_vel(goal_velocities)

        self.orientations = self.orientations_from_velocities(goal_velocities)
        self.velocities[0,:] = np.linalg.norm(goal_velocities, axis = 0)

        self.go_to_pose_and_control()

        #pos_vel_log
        current_position = self.r.get_poses()[:2,:]
        current_velocities = goal_velocities
        self.r.set_velocities(np.arange(self.num_agents), np.zeros((2,self.num_agents)))
        self.r.step()
        current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)
        self.pos_vel_log = np.concatenate([self.pos_vel_log,
                                np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)], axis = 1)

    def orientations_from_velocities(self,velocities):

        speeds = np.linalg.norm(velocities, axis=0, keepdims=True)
        cos_sin = velocities / speeds
        orientations = np.arccos(cos_sin[:1, :]) * np.sign(cos_sin[1:, :])
        return np.nan_to_num(orientations)

    def regularize_vel(self,velocities):

        vel_norm = np.linalg.norm(velocities, axis = 0)

        for i in range(self.num_agents):
            if vel_norm[i] > self.max_vel:
                velocities[:,i] *= self.max_vel/vel_norm[i]

        return velocities



    def go_to_pose_and_control(self):

        no_of_steps = int(self.time_step/self.r.time_step)

        x = self.r.get_poses()
        self.r.step()

        rotation_error = 0.2;

        error  = self.rotation_error(np.reshape(x[2,:],(1,self.num_agents)),self.orientations)
        error_old = np.zeros((1, self.num_agents))
        error_sum = np.zeros((1, self.num_agents))

        while(np.sum(abs(error) < rotation_error) != self.num_agents):

            dxu = np.zeros((2, self.num_agents))

            x = self.r.get_poses()
            error  = self.rotation_error(np.reshape(x[2,:],(1,self.num_agents)),self.orientations)

            angular_vel, error_old, error_sum  = self.pid_controller(error, error_old, error_sum, Kp = self.Kp, Kd =self.Kd, Ki = self.Ki)
            angular_vel = self.normalize_control(angular_vel, self.max_ang_vel)

            dxu[1,:] = angular_vel

            # Create safe input s
            dxu = self.uni_barrier_cert(dxu, x)

            self.r.set_velocities(np.arange(self.num_agents), dxu)
            self.r.step()

        for i in range(no_of_steps):

            x = self.r.get_poses()
            dxu = self.velocities
            # Create safe input s
            dxu = self.uni_barrier_cert(dxu, x)
            self.r.set_velocities(np.arange(self.num_agents), dxu)
            self.r.step()


    def rotation_error(self,current_orientation, desired_orientation):

        error = self.convert_angles_neg_pi_to_pi(desired_orientation) - self.convert_angles_neg_pi_to_pi(current_orientation)

        return self.convert_angles_neg_pi_to_pi(error)

    def convert_angles_neg_pi_to_pi(self,angles):

        N = angles.shape[1]

        while(np.sum(np.logical_and(angles >= 0,angles <= 2 *np.pi)) !=N):

            angles = angles - 2 * np.pi * (angles > 2* np.pi)
            angles = angles + 2 * np.pi * (angles < 0)

        angles = angles - 2 * np.pi * (angles > np.pi)

        return angles

    def pid_controller(self,error, error_old, error_sum, Kp, Kd, Ki):

        error_dot = error - error_old
        error_sum = error + error_sum

        control = Kp * error + Kd * error_dot + Ki * error_sum

        return control, error, error_sum

    def normalize_control(self,control, max_control):

        control[control > max_control] = max_control
        control[control < -max_control] = -max_control

        return control

def main():

    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps

    agents = ChaserAgents(num_agents = ARGS.num_agents, max_vel = ARGS.max_vel, max_acc = ARGS.max_acc, max_ang_vel = ARGS.max_ang_vel, radius = ARGS.radius,
                            time_step = ARGS.time_step, model_params = model_params, log_dir = ARGS.log_dir, show_figure = ARGS.show_figure, Kp = 1, Kd = 0, Ki = 0)
    # ----------------------------------------
    print('Initialzing the agents for 7 steps ')
    print('------------------------------------')
    for i in range(1,8):

        t = time.time()
        agents.move_agents()
        print('Step %d, time = %f' %(i, time.time() - t))

    print('Starting GNN predictions')
    print('===========================================')

    for i in range(8,ARGS.num_time_steps):
         #do some thing
        t = time.time()
        agents.move_agents_GNN()
        print('Step %d, time = %f' %(i, time.time() - t))

    agents.r.call_at_scripts_end()

    time.sleep(3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type = int, default = 10, help = 'number of agents')
    parser.add_argument('--max-vel', type = float, default = 0.2, help = 'max velocity')
    parser.add_argument('--max-ang-vel', type = float, default = np.pi, help = 'max angular velocity')
    parser.add_argument('--time-step', type = float, default = 0.1320, help = 'time step')
    parser.add_argument('--max-acc', type = float, default = 3, help = 'max acceleration')
    parser.add_argument('--radius', type = float, default = 0.75, help = 'initial radius of the formation')
    parser.add_argument('--num-time-steps', type = int, default = 60, help = 'time step')
    parser.add_argument('--show-figure', action='store_true', default = True, help = 'show robotarium figure')
    parser.add_argument('--log-dir', type = str, default = 'log/', help = 'Neural Network training log folder')
    parser.add_argument('--pred-steps', type = int, default = 1, help = 'number of steps the estimator predicts for time series')
    parser.add_argument('--config', type = str, default = 'config_dynamical.json', help = 'model config file')
    parser.add_argument('--batch-size', type = int, default = 1, help = 'batch size')
    #parser.add_argument('--save-data')
    ARGS = parser.parse_args()

    main()
