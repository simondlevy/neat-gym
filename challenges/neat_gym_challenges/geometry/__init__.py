'''
Geometry methods for Maze environment

Adapted from https:#github.com/yaricom/goNEAT_NS/blob/master/experiments/maze/environment.go

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

def distance_point_to_point(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def distance_point_to_line(point, line):
    px, py = point
    la, lb = line
    lax, lay = la
    lbx, lby = lb
    utop = (px - lax) * (lbx - lax) + (py - lay) * (lby - lay)
    ubot = distance_point_to_point(la, lb)
    ubot *= ubot
    if ubot == 0:
        return 0
    u = utop / ubot
    if u < 0 or u > 1:
        d1 = distance_point_to_point(la, point)
        d2 = distance_point_to_point(lb, point)
        return min(d1, d2)
    newpoint = lax + u * (lbx - lax), lay + u * (lby - lay)
    return distance_point_to_point(point, newpoint)

def rotate(p, q, angle):
    '''
    Rotates point p around point q point by given angle in degrees
    '''
    px,py = p
    qx,qy = q

    px -= qx
    py -= qy

    ox, oy = px, py

    rad = np.radians(angle)
    px = np.cos(rad) * ox - np.sin(rad) * oy
    py = np.sin(rad) * ox + np.cos(rad) * oy

    px += qx
    py += qy

    return px,py

def intersection(line1, line2):

    A, B = line1
    C, D = line2

    AX, AY = A
    BX, BY = B
    CX, CY = C
    DX, DY = D

    rTop = (AY - CY) * (DX - CX) - (AX - CX) * (DY - CY)
    rBot = (BX - AX) * (DY - CY) - (BY - AY) * (DX - CX)

    sTop = (AY - CY) * (BX - AX) - (AX - CX) * (BY - AY)
    sBot = (BX - AX) * (DY - CY) - (BY - AY) * (DX - CX)

    if rBot == 0 or sBot == 0: 
        # lines are parallel
        return None

    r = rTop / rBot
    s = sTop / sBot

    if r > 0 and r < 1 and s > 0 and s < 1: 
        ptx = AX + r * (BX - AX)
        pty = AY + r * (BY - AY)
        return ptx,pty

    return None

def arctan_degrees(point):
    ang = np.degrees(np.arctan2(point[1], point[0]))
    return ang + 360 if ang < 0 else ang # support lower quadrants (3 and 4)
