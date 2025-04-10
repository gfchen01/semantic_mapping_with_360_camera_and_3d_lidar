from pytorch3d.ops import box3d_overlap
import torch
import time
import numpy as np

def get_corners_from_box3d_torch(centers, extents, angles):
    corners = torch.zeros(centers.shape[0], 8, 3)
    for i in range(centers.shape[0]):
        center = centers[i]
        extent = extents[i]
        angle = angles[i]

        # Get rotation matrix
        R = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])

        # Get corners
        corners[i, 0] = center + R @ torch.tensor([extent[0], extent[1], extent[2]])
        corners[i, 1] = center + R @ torch.tensor([extent[0], extent[1], -extent[2]])
        corners[i, 2] = center + R @ torch.tensor([extent[0], -extent[1], extent[2]])
        corners[i, 3] = center + R @ torch.tensor([extent[0], -extent[1], -extent[2]])
        corners[i, 4] = center + R @ torch.tensor([-extent[0], extent[1], extent[2]])
        corners[i, 5] = center + R @ torch.tensor([-extent[0], extent[1], -extent[2]])
        corners[i, 6] = center + R @ torch.tensor([-extent[0], -extent[1], extent[2]])
        corners[i, 7] = center + R @ torch.tensor([-extent[0], -extent[1], -extent[2]])
    
    return corners


def generate_random_box3d_torch(n):
    center = torch.rand(n, 3)
    extent = torch.rand(n, 3) * 2.0
    angle = torch.rand(n) * 3.0
    corners = get_corners_from_box3d(center, extent, angle)

    return corners

def get_corners_from_box3d(centers, extents, angles):
    corners = np.zeros((centers.shape[0], 8, 3))
    for i in range(centers.shape[0]):
        center = centers[i]
        extent = extents[i]
        angle = angles[i]

        # Get rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Get corners
        corners[i, 0] = center + np.dot(R, np.array([extent[0], extent[1], extent[2]]))
        corners[i, 1] = center + np.dot(R, np.array([extent[0], extent[1], -extent[2]]))
        corners[i, 2] = center + np.dot(R, np.array([extent[0], -extent[1], extent[2]]))
        corners[i, 3] = center + np.dot(R, np.array([extent[0], -extent[1], -extent[2]]))
        corners[i, 4] = center + np.dot(R, np.array([-extent[0], extent[1], extent[2]]))
        corners[i, 5] = center + np.dot(R, np.array([-extent[0], extent[1], -extent[2]]))
        corners[i, 6] = center + np.dot(R, np.array([-extent[0], -extent[1], extent[2]]))
        corners[i, 7] = center + np.dot(R, np.array([-extent[0], -extent[1], -extent[2]]))
    
    return corners

def generate_random_box3d(n):
    center = np.random.rand(n, 3)
    extent = np.random.rand(n, 3) * 2.0
    angle = np.random.rand(n) * 3.0
    corners = get_corners_from_box3d(center, extent, angle)

    return corners

from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def test_box3d_overlap_torch():
    n = 1

    boxes1 = generate_random_box3d_torch(n)
    boxes2 = generate_random_box3d_torch(n)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)

    start = time.time()
    intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)
    print("Time taken: ", time.time() - start, "s, for n = ", n)
    print(iou_3d.shape)

def test_box3d_overlap():
    n = 1

    boxes1 = generate_random_box3d(n)
    boxes2 = generate_random_box3d(n)

    start = time.time()
    for i in range(n):
        print(boxes1[i], boxes2[i])
        iou, iou_2d = box3d_iou(boxes1[i], boxes2[i])
        print(iou, iou_2d)
    print("Time taken: ", time.time() - start, "s, for n = ", n)
    

if __name__ == "__main__":
    test_box3d_overlap()