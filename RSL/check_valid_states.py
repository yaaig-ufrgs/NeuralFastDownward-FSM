from math import sqrt

class ValidStatesValidator:
    def __init__(self, problem):
        self.problem = problem
        self.npuzzle, self.size = self.fill_npuzzle_matrix()

    def fill_npuzzle_matrix(self):
        max_tile = 0
        for atom in [str(atom) for atom in self.problem.as_atoms()]:
            atom_split = atom.split('(')
            arg_split = [x.strip(')') for x in atom_split[1].split(',')]
            if atom_split[0] == "at":
                tile = int(arg_split[0][2:])
                if tile > max_tile:
                    max_tile = tile
        max_tile += 1
        size = int(sqrt(max_tile))
        return ([[None]*size for i in range(size)], max_tile)

    def is_valid(self, state):
        if len(state) > self.size:
            return False

        self.parse_npuzzle_state(state)
        if self.npuzzle_matrix_has_none():
            return False

        #self.npuzzle = [[8, 1, 2],[0, 4, 3],[7, 6, 5]] # unsolvable example
        #self.npuzzle = [[1, 8, 2],[0, 4, 3],[7, 6, 5]] # solvable example
        inversions = self.count_inversions([j for sub in self.npuzzle for j in sub])
        self.clear_npuzzle_matrix()

        # It is not possible to solve an npuzzle instance if the number of
        # inversions is odd in the input size.
        # https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
        return inversions % 2 == 0
        
    def count_inversions(self, arr):
        inv_count = 0
        empty_value = 0
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if arr[j] != empty_value and arr[i] != empty_value and arr[i] > arr[j]:
                    inv_count += 1
        return inv_count

    def parse_npuzzle_state(self, state):
        empty_inserted = False
        for atom in state:
            atom_split = atom.split('(')
            arg_split = [x.strip(')') for x in atom_split[1].split(',')]
            if atom_split[0] == "at":
                # at(t_n, p_x_y)
                coords = arg_split[1].split('_')
                x, y = int(coords[1]), int(coords[2])
                tile = int(arg_split[0][2:])
                self.npuzzle[x-1][y-1] = tile

            elif atom_split[0] == "empty":
                empty_inserted = True
                # empty(p_x_y)
                coords = arg_split[0].split('_')
                x, y = int(coords[1]), int(coords[2])
                self.npuzzle[x-1][y-1] = 0

        # Special goal case
        if empty_inserted is False:
            last_idx = int(sqrt(len(state)+1))-1
            self.npuzzle[last_idx][last_idx] = 0

    def npuzzle_matrix_has_none(self):
        for i in range(len(self.npuzzle)):
            for j in range(len(self.npuzzle)):
                if self.npuzzle[i][j] == None:
                    return True
        return False

    def clear_npuzzle_matrix(self):
        size = int(sqrt(self.size))
        self.npuzzle = [[None for i in range(size)] for j in range(size) ]
