from scipy.spatial import KDTree
import numpy as np


def find_closest_indices_kdtree(x, y, query_x, query_y):
    """
    Find the indices of the points in arrays `x` and `y` that are closest to
    the given query points, whether single or multiple, using a KD-Tree.

    Parameters:
    - x (array-like): Array of x-coordinates of the dataset.
    - y (array-like): Array of y-coordinates of the dataset.
    - query_x (float or array-like): x-coordinate(s) for query point(s).
    - query_y (float or array-like): y-coordinate(s) for query point(s).

    Returns:
    - list of int or int: Index/Indices of the closest point(s) in `x` and `y`.
    """
    # Combine x and y into a 2D array of dataset points
    points = np.vstack((x, y)).T

    # Handle single point query by converting it to an array with one element
    if np.isscalar(query_x) and np.isscalar(query_y):
        query_points = np.array([[query_x, query_y]])
        single_query = True
    else:
        query_points = np.vstack((query_x, query_y)).T
        single_query = False

    # Create a KD-Tree from the dataset points
    tree = KDTree(points)

    # Query the KD-Tree for the nearest neighbors to the query points
    _, closest_indices = tree.query(query_points)

    # Return a single integer if it was a single query, or a list if it was multiple
    return closest_indices[0] if single_query else closest_indices


from scipy.spatial import KDTree
import numpy as np


class FaultGeometryKDTree:
    def __init__(self, x, y):
        """
        Initializes a KDTree for efficient nearest-neighbor queries for the given fault geometry.

        Parameters:
        - x (array-like): Array of x-coordinates of the fault geometry.
        - y (array-like): Array of y-coordinates of the fault geometry.
        """
        # Combine x and y into a 2D array of dataset points
        self.points = np.vstack((x, y)).T
        # Create a KD-Tree from the dataset points
        self.tree = KDTree(self.points)

    def find_closest_indices(self, query_x, query_y):
        """
        Find the indices of the points in the KDTree that are closest to
        the given query points, whether single or multiple.

        Parameters:
        - query_x (float or array-like): x-coordinate(s) for query point(s).
        - query_y (float or array-like): y-coordinate(s) for query point(s).

        Returns:
        - list of int or int: Index/Indices of the closest point(s) in the KDTree.
        """
        # Handle single point query by converting it to an array with one element
        if np.isscalar(query_x) and np.isscalar(query_y):
            query_points = np.array([[query_x, query_y]])
            single_query = True
        else:
            query_points = np.vstack((query_x, query_y)).T
            single_query = False

        # Query the KD-Tree for the nearest neighbors to the query points
        _, closest_indices = self.tree.query(query_points)

        # Return a single integer if it was a single query, or a list if it was multiple
        return closest_indices[0] if single_query else closest_indices
