#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq



class TSPSolver:
    def __init__( self, gui_view ):
        self._scenario = None

    def setupWithScenario( self, scenario ):
        self._scenario = scenario


    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour
        </summary>
        <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (
not counting initial BSSF estimate)</returns> '''
    def defaultRandomTour( self, start_time, time_allowance=60.0 ):

        results = {}


        start_time = time.time()

        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        while not foundTour:
            # create a random permutation
            perm = np.random.permutation( ncities )

            #for i in range( ncities ):
                #swap = i
                #while swap == i:
                    #swap = np.random.randint(ncities)
                #temp = perm[i]
                #perm[i] = perm[swap]
                #perm[swap] = temp

            route = []

            # Now build the route using the random permutation
            for i in range( ncities ):
                route.append( cities[ perm[i] ] )

            bssf = TSPSolution(route)
            #bssf_cost = bssf.cost()
            #count++;
            count += 1

            #if costOfBssf() < float('inf'):
            if bssf.costOfRoute() < np.inf:
                # Found a valid route
                foundTour = True
        #} while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
        #timer.Stop();

        results['cost'] = bssf.costOfRoute() #costOfBssf().ToString();                          // load results array
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf

       # return results;
        return results



    def greedy( self, start_time, time_allowance=60.0 ):
        pass

    def branchAndBound( self, start_time, time_allowance=60.0 ):
        # Declare results dictionary
        results = {}
        # Start the timer
        start_time = time.time()
        # Get the cities
        cities = self._scenario.getCities()
        # Count the cities
        ncities = len(cities)
        # Create a priority queue
        pq = []
        # Create an initial partial solution
        # Create non-reduced TSP matrix
        M = self.getTSPMatrix(cities)
        # Set previous bound to 0
        prev_bound = 0
        # Reduce the matrix
        bound = self.reduceMatrix(M, prev_bound)
        # Insert the initial partial solution into the priority queue
        partials = {bound:{'M':M, 'route':[]}}
        heapq.heappush(pq, bound)  # Empty route

        # Iterate while time is not expired and the priority queue is not empty
        while time_allowance > time.time() - start_time and len(pq) != 0:
            # Pop parent partial solution off queue
            parent_bound = pq.pop()
            parent_M = partials[parent_bound]['M']
            parent_route = partials[parent_bound]['route']
            # Generate children

            # Create a child for each city that is not in the tour
            # Count each child state generated
            # for each city not in tour
            # Create child
            # If bound of child is less than bssf
            pass
            # Check for a solution
            # if len(route) == ncities:
                # If that solution improves bssf
                # bssf = TSPSolution(route)
                # Trim the priority queue
                # else keep previous bssf


        pass

    def getTSPMatrix(self, cities):
        """
        Create a matrix that represents the cities relationship to each other
        :param cities: List of City objects
        :return: np.matrix
        """
        ncities = len(cities)   # Get the number of cities
        M = np.zeros((ncities, ncities))  # Initialize a matrix of zeros
        # For each city
        for row in range(ncities):
            # And for every column
            for col in range(ncities):
                # Get the cost to the other city
                to_city = cities[col]       # The city we are going to
                from_city = cities[row]     # The city we are leaving
                M[row][col] = from_city.costTo(to_city)
        # Return the matrix
        return M

    def reduceMatrix(self, M, prev_bound):
        """
        Make it so there is a zero in each row and each col. Also calculate the bound
        Reduce a matrix
        :param m: A np.matrix
        :return: A reduced matrix, bound int
        """
        bound = prev_bound  # Initialize bound to be previous lower bound
        ncities = len(M)
        # Perform a row reduction
        # Iterate over each row
        for i in range(ncities):
            min_value = np.min(M[i])
            M[i] -= min_value
            # Add the subtracted value to the bound
            bound += min_value

        # Perform a col reduction
        # Iterate over each col
        for j in range(ncities):
            # Find the value and index of the smallest element in the row
            min_value = np.min(M[:, j])
            M[:, j] -= min_value
            bound += min_value

        return bound


    def generateChild(self, dest):
        """
        Get a child solution based on what city we are going to next
        :param dest: The city we are going to next
        :return: Solution object (may be a partial solution)
        """
        pass

    def fancy( self, start_time, time_allowance=60.0 ):
        pass



