#!/usr/bin/env python
# coding: gbk

########################################################################
# Author: Feng Ruohang
# Create: 2014/10/02 13:39
# Digest: Computate the convex hull of a given point list
########################################################################


direction = lambda m: (m[2][0] - m[0][0]) * (m[1][1] - m[0][1]) - (m[1][0] - m[0][0]) * (m[2][1] - m[0][1])
'''
    A Quick Side_check version Using Lambda expression
    Input:  Given a list of three point : m should like [(p_x,p_y), (q_x,q_y), (r_x,r_y)]
    Output: Return a Number to indicate whether r on the right side of vector(PQ).
    Positive means r is on the right side of vector(PQ).
    This is negative of cross product of PQ and PR: Defined by:(Qx-Px)(Ry-Py)-(Rx-Px)(Qy-Py)
    Which 'negative' indicate PR is clockwise to PQ, equivalent to R is on the right side of PQ
'''


def convex_hull(point_list):
  '''
  Input:  Given a point List: A List of Truple (x,y)
  Output: Return a point list: A List of Truple (x,y) which is CONVEX HULL of input
  For the sake of effeciency, There is no error check mechanism here. Please catch outside
  '''
  n = len(point_list)  # Total Length
  point_list.sort()

  # Valid Check:
  if n < 3:
    return len(point_list)

  # Building Upper Hull: Initialized with first two point
  upper_hull = point_list[0:1]
  for i in range(2, n):
    upper_hull.append(point_list[i])
    while len(upper_hull) >= 3 and not direction(upper_hull[-3:]):
      del upper_hull[-2]

  # Building Lower Hull: Initialized with last two point
  lower_hull = [point_list[-1], point_list[-2]]
  for i in range(n - 3, -1, -1):  # From the i-3th to the first point
    lower_hull.append(point_list[i])
    while len(lower_hull) >= 3 and not direction(lower_hull[-3:]):
      del lower_hull[-2]
  upper_hull.extend(lower_hull[1:-1])
  return upper_hull


# ========Unit Test:
if __name__ == '__main__':
  test_data = [(i, i ** 2) for i in range(1, 100)]
  result = convex_hull(test_data)
  print result

