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
        # Create the tour
        tour = []
        # Create a priority queue
        pq =[]

        # Create non-reduced TSP matrix
        m = self.getTSPMatrix(cities)

        # Reduce the matrix
        self.reduceMatrix(m)

        # Create child states

        # for each city that is not in tour (the partial solution)
        # Count each child state generated
        # child = self.getChild(dest)

        # Check if child bound is better than bssf
        # Insert child into priority queue
        pq.heappush()

        # Iterate while time is not expired and the priority queue is not empty
        while time_allowance < time.time() - start_time and pq.count() != 0:
            # Pop parent off queue
            parent = pq.pop()
            # Add the parent to the tour
            tour.append(parent)
            # Create a child for each city that is not in the tour
            # Count each child state generated
            # for each city not in tour
            # Create child
            # If bound of child is less than bssf
            pq.heappush()

            # Check for a solution
            if len(tour) == ncities:
                # If that solution improves bssf
                bssf = TSPSolution(tour)
                # Trim the priority queue
                # else keep previous bssf


        pass

    def getTSPMatrix(self, cities):
        """
        Create a matrix that represents the cities relationship to each other
        :param cities: List of City objects
        :return: np.matrix
        """
        pass

    def reduceMatrix(self, m):
        """
        Reduce a matrix
        :param m: A np.matrix
        :return: A reduced matrix, bound int
        """
        pass

    def getChild(self, dest):
        """
        Get a child solution based on what city we are going to next
        :param dest: The city we are going to next
        :return: Solution object (may be a partial solution)
        """
        pass

    def fancy( self, start_time, time_allowance=60.0 ):
        pass



