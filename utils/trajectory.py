# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import numpy as np
import torch


def generate_seed(scale, viewangle):
    # World 2 Camera
    #### rotate x,y
    render_poses = [np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])]
    ang = 5
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.eye(3),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))) # Turn left
        posetemp[:3,3:4] = np.array([0,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    for i,j in zip([-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(-3*ang/180*np.pi), 0, np.sin(-3*ang/180*np.pi)], [0, 1, 0], [-np.sin(-3*ang/180*np.pi), 0, np.cos(-3*ang/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
        posetemp[:3,3:4] = np.array([1,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(3*ang/180*np.pi), 0, np.sin(3*ang/180*np.pi)], [0, 1, 0], [-np.sin(3*ang/180*np.pi), 0, np.cos(3*ang/180*np.pi)]]), 
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
        posetemp[:3,3:4] = np.array([-1,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.eye(3), 
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([0,0,1]).reshape(3,1) # * scale # Transition vector   
    #     render_poses.append(posetemp)


    rot_cam=viewangle/3
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))) # Turn left
        posetemp[:3,3:4] = np.array([0,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)

    for i,j in zip([-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(-3*ang/180*np.pi), 0, np.sin(-3*ang/180*np.pi)], [0, 1, 0], [-np.sin(-3*ang/180*np.pi), 0, np.cos(-3*ang/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([0,0,1]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(3*ang/180*np.pi), 0, np.sin(3*ang/180*np.pi)], [0, 1, 0], [-np.sin(3*ang/180*np.pi), 0, np.cos(3*ang/180*np.pi)]]), 
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([0,0,-1]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([1,0,0]).reshape(3,1) # * scale # Transition vector   
    #     render_poses.append(posetemp)


    rot_cam=viewangle*2/3
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))) # Turn left
        posetemp[:3,3:4] = np.array([0,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)

    for i,j in zip([-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(-3*ang/180*np.pi), 0, np.sin(-3*ang/180*np.pi)], [0, 1, 0], [-np.sin(-3*ang/180*np.pi), 0, np.cos(-3*ang/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([-1,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(3*ang/180*np.pi), 0, np.sin(3*ang/180*np.pi)], [0, 1, 0], [-np.sin(3*ang/180*np.pi), 0, np.cos(3*ang/180*np.pi)]]), 
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([1,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([0,0,-1]).reshape(3,1) # * scale # Transition vector   
    #     render_poses.append(posetemp)

    rot_cam=viewangle
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))) # Turn left
        posetemp[:3,3:4] = np.array([0,0,0]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)

    for i,j in zip([-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(-3*ang/180*np.pi), 0, np.sin(-3*ang/180*np.pi)], [0, 1, 0], [-np.sin(-3*ang/180*np.pi), 0, np.cos(-3*ang/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([0,0,-1]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,ang,ang,ang,ang,0,-ang,-ang,-ang,-ang,-ang,0,0,0,0]): 
        th, phi = i/180*np.pi, j/180*np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
                                    np.matmul(np.array([[np.cos(3*ang/180*np.pi), 0, np.sin(3*ang/180*np.pi)], [0, 1, 0], [-np.sin(3*ang/180*np.pi), 0, np.cos(3*ang/180*np.pi)]]), 
                                    np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]))))
        posetemp[:3,3:4] = np.array([0,0,1]).reshape(3,1) # * scale # Transition vector   
        render_poses.append(posetemp)
    
    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]): 
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([-1,0,0]).reshape(3,1) # * scale # Transition vector   
    #     render_poses.append(posetemp)

    render_poses.append(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    render_poses = np.stack(render_poses, axis=0)

    return render_poses


def generate_seed_360(viewangle, n_views):
    N = n_views
    render_poses = np.zeros((N, 3, 4))
    new_render_poses = np.zeros((N, 3, 4))
    thetalist = []
    philist = []
    for i in range(N):
        th = (viewangle/N)*i/180*np.pi
        phi = 0
        thetalist.append(th*180/np.pi)
        philist.append(phi*180/np.pi)
        
        q = quaternion_from_euler(phi, th)
        render_poses[i,:3,:3] = quaternion_to_rotmat(q)
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0

        new_render_poses[i,:3,:3] = render_poses[i,:3,:3].T
        new_render_poses[i,:3,3:4] = np.array([0,0,20]).reshape(3,1)
        print(new_render_poses[i])
    return new_render_poses

def generate_seed_perturb(perturb = 0.1):
    render_poses = np.zeros((5, 3, 4))
    new_render_poses = np.zeros((5, 4, 4))
    th = 0
    phi = 0
    for i in range(5):
        q = quaternion_from_euler(phi, th)
        render_poses[i,:3,:3] = quaternion_to_rotmat(q)

        if i % 5 == 0:
            render_poses[i,:3,3:4] = np.array([0, 0, 0]).reshape(3,1)
        elif i % 5 == 1:
            render_poses[i,:3,3:4] = np.array([perturb, 0, 0]).reshape(3,1)
        elif i % 5 == 2:
            render_poses[i,:3,3:4] = np.array([0, perturb, 0]).reshape(3,1)
        elif i % 5 == 3:
            render_poses[i,:3,3:4] = np.array([-perturb, 0, 0]).reshape(3,1)
        elif i % 5 == 4:
            render_poses[i,:3,3:4] = np.array([0, -perturb, 0]).reshape(3,1)
        
        new_render_poses[i,:3,:3] = render_poses[i,:3,:3].T
        new_render_poses[i,:3,3:4] = -np.matmul(render_poses[i,:3,:3].T, render_poses[i,:3,3:4])
        new_render_poses[i,3:4,3:4] = 1.0
        
    return new_render_poses


def generate_seed_360_half(viewangle, n_views):
    N = n_views // 2
    halfangle = viewangle / 2
    render_poses = np.zeros((N*2, 3, 4))
    for i in range(N): 
        th = (halfangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    for i in range(N):
        th = -(halfangle/N)*i/180*np.pi
        render_poses[i+N,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i+N,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    return render_poses


def generate_seed_preset():
    degsum = 60 
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    # 4 + 3 + 4 + 3 + 4 + 3 = 21
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(-22.5,-22.5,7), np.linspace(22.5,22.5,7))) # 7 + 7 + 7 = 21
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_newpreset():
    degsum = 60
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(22.5,22.5,7)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_horizon():
    movement = np.linspace(0, 5, 11)
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        
        render_poses[i,:3,:3] = np.eye(3)
        render_poses[i,:3,3:4] = np.array([[-movement[i]], [0], [0]])

    return render_poses


def generate_seed_backward():
    movement = np.linspace(0, 5, 11)
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i,:3,:3] = np.eye(3)
        render_poses[i,:3,3:4] = np.array([[0], [0], [movement[i]]])
    return render_poses


def generate_seed_arc():
    degree = 5
    # thlist = np.array([degree, 0, 0, 0, -degree])
    thlist = np.arange(0, degree, 5) + np.arange(0, -degree, 5)[1:]
    phi = 0

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        d = 4.3
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[0,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_hemisphere_(degree, nviews):
    # thlist = np.array([degree, 0, 0, 0, -degree])
    # philist = np.array([0, -degree, 0, degree, 0])
    thlist = degree * np.sin(np.linspace(0, 2*np.pi, nviews))
    philist = degree * np.cos(np.linspace(0, 2*np.pi, nviews))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = 4.3
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[0,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses


def generate_seed_nothing():
    degree = 5
    thlist = np.array([0])
    philist = np.array([0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = 4.3
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[0,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_lookaround():
    degsum = 60 
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(22.5,22.5,7), np.linspace(-22.5,-22.5,7)))
    assert len(thlist) == len(philist) # 21

    render_poses = []
    # up / left --> right
    thlist = np.linspace(-degsum, degsum, 2*degsum+1)
    for i in range(len(thlist)):
        render_pose = np.zeros((3,4))
        th = thlist[i]
        phi = 22.5
        
        render_pose[:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_pose[:3,3:4] = np.zeros((3,1))
        render_poses.append(render_pose)
    
    # right / up --> center
    phlist = np.linspace(22.5, 0, 23)
    # Exclude first frame (same as last frame before)
    phlist = phlist[1:]
    for i in range(len(phlist)):
        render_pose = np.zeros((3,4))
        th = degsum
        phi = phlist[i]
        
        render_pose[:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_pose[:3,3:4] = np.zeros((3,1))
        render_poses.append(render_pose)

    # center / right --> left
    thlist = np.linspace(degsum, -degsum, 2*degsum+1)
    thlist = thlist[1:]
    for i in range(len(thlist)):
        render_pose = np.zeros((3,4))
        th = thlist[i]
        phi = 0
        
        render_pose[:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_pose[:3,3:4] = np.zeros((3,1))
        render_poses.append(render_pose)

    # left / center --> down
    phlist = np.linspace(0, -22.5, 23)
    phlist = phlist[1:]
    for i in range(len(phlist)):
        render_pose = np.zeros((3,4))
        th = -degsum
        phi = phlist[i]
        
        render_pose[:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_pose[:3,3:4] = np.zeros((3,1))
        render_poses.append(render_pose)


    thlist = np.linspace(-degsum, degsum, 2*degsum+1)
    for i in range(len(thlist)):
        render_pose = np.zeros((3,4))
        th = thlist[i]
        phi = -22.5
        
        render_pose[:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_pose[:3,3:4] = np.zeros((3,1))
        render_poses.append(render_pose)

    return render_poses


def generate_seed_lookdown():
    degsum = 60 
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(-22.5,-22.5,7)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_back(distance=5):
    movement = np.linspace(0, distance, 101)
    render_poses = [] # np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_pose = np.zeros((3,4))
        render_pose[:3,:3] = np.eye(3)
        render_pose[:3,3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    movement = np.linspace(distance, 0, 101)
    movement = movement[1:]
    for i in range(len(movement)):
        render_pose = np.zeros((3,4))
        render_pose[:3,:3] = np.eye(3)
        render_pose[:3,3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    return render_poses


def generate_seed_llff(degree, nviews, round=4, d=2.3):
    assert round%4==0
    # thlist = np.array([degree, 0, 0, 0, -degree])
    # philist = np.array([0, -degree, 0, degree, 0])
    # d = 2.3
    thlist = degree * np.sin(np.linspace(0, 2*np.pi*round, nviews))
    philist = degree * np.cos(np.linspace(0, 2*np.pi*round, nviews))
    zlist = d/15 * np.sin(np.linspace(0, 2*np.pi*round//4, nviews))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        z = zlist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, -z+d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), -z+d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses


def generate_seed_headbanging(maxdeg, nviews_per_round, round=3, fullround=1):
    radius = np.concatenate((np.linspace(0, maxdeg, nviews_per_round*round), maxdeg*np.ones(nviews_per_round*fullround), np.linspace(maxdeg, 0, nviews_per_round*round)))
    thlist  = 2.66*radius * np.sin(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    philist = radius * np.cos(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_headbanging_circle(maxdeg, nviews_per_round, round=3, fullround=1):
    radius = np.concatenate((np.linspace(0, maxdeg, nviews_per_round*round), maxdeg*np.ones(nviews_per_round*fullround), np.linspace(maxdeg, 0, nviews_per_round*round)))
    thlist  = 2.66*radius * np.sin(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    philist = radius * np.cos(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses

def generate_seed_rotateshow(radius, num_frames):
    
    render_poses = np.zeros((num_frames, 3, 4))
    new_render_poses = np.zeros((num_frames, 3, 4))

    for i in range(num_frames):
        theta = 360 * i / num_frames
        phi = 0
        
        d = radius * np.sin(2 * np.pi * i * 2 / num_frames)

        R = np.matmul(np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]), np.array([[np.cos(theta/180*np.pi), 0, -np.sin(theta/180*np.pi)], [0, 1, 0], [np.sin(theta/180*np.pi), 0, np.cos(theta/180*np.pi)]]))
        T = np.array([0, 0, d]).reshape(3,1)
        theta = theta/180*np.pi
        phi = phi/180*np.pi

        q = quaternion_from_euler(phi, theta)
        render_poses[i,:3,:3] = quaternion_to_rotmat(q)
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0

        new_render_poses[i,:3,:3] = render_poses[i,:3,:3].T
        render_poses[i]=np.hstack([new_render_poses[i,:3,:3], T])
        
    return render_poses


def get_pcdGenPoses(pcdgenpath, argdict={}):
    if pcdgenpath == 'back_and_forth':
        render_poses = generate_seed_back(1)
    elif pcdgenpath == 'llff':
        render_poses = generate_seed_llff(5, 400, round=4, d=2)
    elif pcdgenpath == 'headbanging':
        render_poses = generate_seed_headbanging(maxdeg=15, nviews_per_round=180, round=2, fullround=0)
    elif pcdgenpath == 'rotate360':
        render_poses = generate_seed_360(360, 360)
    elif pcdgenpath == 'rotateshow':
        render_poses = generate_seed_rotateshow(0.8, 540)
    else:
        raise("Invalid pcdgenpath")
    return render_poses


def quaternion_from_euler(phi, theta):
    # transform euler angles to quaternion
    # rotate around X 
    q_x = np.array([np.cos(-phi / 2), np.sin(-phi / 2), 0, 0])
    # rotate around Y
    q_y = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0])
    q = quaternion_multiply(q_x, q_y)
    return q

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_to_rotmat(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    return R


if __name__ == '__main__':
    main()