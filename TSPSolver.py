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
import copy



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
        # Child count
        child_count = 0
        # BSSF update count
        bssf_count = 0
        max_states = 0
        pruned_count = 0
        # Initialize the best solution so far
        bssf = PartialSolution(np.zeros((ncities, ncities)), math.inf, [])  # Initialize a matrix of zeros

        # Create an initial partial solution
        # Create non-reduced TSP matrix
        M = self.getTSPMatrix(cities)
        # Set previous bound to 0
        prev_bound = 0
        # Reduce the matrix
        bound = self.reduceMatrix(M, prev_bound)
        # Insert the initial partial solution into the priority queue
        partials = {}
        partials[bound] = PartialSolution(M, bound, [0])
        heapq.heappush(pq, bound)  # Empty route

        # Iterate while time is not expired and the priority queue is not empty
        while time_allowance > time.time() - start_time and len(pq) != 0:
            # Pop parent partial solution off queue
            # Get other parent details
            parent = partials[heapq.heappop(pq)]
            # Generate n children
            # Get the row of the city we are currently at
            i = parent.route[len(parent.route) - 1]
            for j in range(ncities):
                # Check if edge exists
                if parent.M[i][j] != math.inf:
                    child_count += 1  # Increment the number of children we have created
                    child = self.generateChild(parent, i, j)
                    # Check if child's bound is less than the current BSSF bound
                    if child.bound < bssf.bound:
                        # Add the child to the priority queue
                        heapq.heappush(pq, child.bound)
                        partials[child.bound] = child
                        # Check if the child is a solution
                        if len(child.route) == ncities:
                            bssf_count += 1
                            bssf = child
                            pq_len = len(pq)
                            if max_states < pq_len:
                                max_states = pq_len
                            # Trim the priority queue
                            self.trim(pq, bssf.bound)
                            pruned_count += pq_len - len(pq)

        # Initialize results array
        c = []

        for x in range(len(bssf.route)):
            c.append(cities[bssf.route[x]])

        bssf = TSPSolution(c)
        results['cost'] = bssf.costOfRoute()
        results['time'] = time.time() - start_time
        results['count'] = bssf_count
        results['soln'] = bssf

        print('Total # of states created: {}' .format(child_count))
        print('Total # of states pruned: {}' .format(pruned_count))
        print('Max # of states at a time: {}' .format(max_states))

        return results

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
            if min_value == math.inf:
                continue
            M[i] -= min_value
            # Add the subtracted value to the bound
            bound += min_value

        # Perform a col reduction
        # Iterate over each col
        for j in range(ncities):
            # Find the value and index of the smallest element in the row
            min_value = np.min(M[:, j])
            if min_value == math.inf:
                continue
            M[:, j] -= min_value
            bound += min_value

        return bound

    def generateChild(self, parent, i, j):
        """
        Get a child solution based on what city we are going to next
        :param parent:
        :param i
        :param j
        :return: child PartialSolution object
        """
        child = copy.deepcopy(parent)
        # Look at the value in M to get the child's bound
        child.bound += child.M[i][j]

        # Eliminate the row i and col j from further consideration
        x = np.ones(len(parent.M))*math.inf
        child.M[i] = x
        child.M[:, j] = x
        # Take out arrival edge
        child.M[j][i] = math.inf

        # Reduce the child matrix
        child.bound = self.reduceMatrix(child.M, child.bound)

        # Add destination city to route
        child.route.append(j)

        return child

    def trim(self, pq, bssf_bound):
        """
        Get rid of elements that have a bound that is higher than the bound of the bssf
        :param pq: Priority Queue
        :param bssf_bound: The bound that we are comparing the elements of the pq
        :return: nothing
        """
        for x in range(len(pq)):
            if pq[x] >= bssf_bound:
                del pq[x:]
                break



    def fancy( self, start_time, time_allowance=60.0 ):
        pass



