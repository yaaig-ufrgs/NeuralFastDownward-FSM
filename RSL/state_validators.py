from math import sqrt
import tarski

class blocks_state_validator:
    def __init__(self, atoms: tarski.util.SymbolIndex):
        self.nodes = {}
        for atom in [str(a) for a in atoms]:
            if "clear(" in atom:
                v = atom.split("clear(")[1].split(")")[0]
                self.nodes[v] = ([], [])

    def is_valid(self, state: set):
        for v in self.nodes:
            self.nodes[v] = ([], [])
        holding = False
        for atom in state:
            if "on(" in atom:
                src, dst = atom.split("on(")[1].split(")")[0].split(",")
                assert src in self.nodes and dst in self.nodes
                if self.nodes[src][0] != [] or self.nodes[dst][1] != []:
                    return False
                self.nodes[src][0].append(dst)
                self.nodes[dst][1].append(src)
            if "holding(" in atom:
                if holding:
                    return False
                holding = True
        for atom in state:
            if "ontable(" in atom:
                v = atom.split("ontable(")[1].split(")")[0]
                if self.nodes[v][0] != []:
                    return False
            elif "clear(" in atom:
                v = atom.split("clear(")[1].split(")")[0]
                if self.nodes[v][1] != []:
                    return False
            elif "handempty(" in atom:
                if "holding(" in state:
                    return False
            elif "holding(" in atom:
                v = atom.split("holding(")[1].split(")")[0]
                if self.nodes[v][0] != [] or self.nodes[v][1] != []:
                    return False
        return True

class npuzzle_state_validator:
    def __init__(self, init_atoms: tarski.model.Model):
        self.init_atoms = init_atoms
        self.npuzzle, self.size = self.fill_npuzzle_matrix()

    def fill_npuzzle_matrix(self):
        max_tile = 0
        for atom in [str(atom) for atom in self.init_atoms.as_atoms()]:
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

        inversions = self.count_inversions([j for sub in self.npuzzle for j in sub])
        self.clear_npuzzle_matrix()

        # It is not possible to solve an npuzzle instance if the number of
        # inversions is odd in the input size.
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

class scanalyzer_state_validator:
    def __init__(self):
        pass

    def is_valid(self, state: set):
        on = []
        for atom in state:
            if "on(" in atom:
                c, _ = atom.split("on(")[1].split(")")[0].split(",")
                if c in on:
                    return False
                on.append(c)
        return True

class transport_state_validator:
    def __init__(self):
        pass

    def is_valid(self, state: set):
        location, capacity = [], []
        for atom in state:
            if "at(" in atom or "in(" in atom:
                o, _ = atom.split("at(" if "at(" in atom else "in(")[1].split(")")[0].split(",")
                if o in location:
                    return False
                location.append(o)
            elif "capacity(" in atom:
                v, _ = atom.split("capacity(")[1].split(")")[0].split(",")
                if v in capacity:
                    return False
                capacity.append(v)
        return True

class visitall_state_validator:
    def __init__(self, init_atoms: tarski.model.Model):
        self.nodes = {}
        self.visited = {}
        for atom in [str(a) for a in init_atoms.as_atoms()]:
            if "connected(" in atom:
                src, dst = atom.split("connected(")[1].split(")")[0].split(",")
                if src not in self.nodes:
                    self.nodes[src] = []
                if dst not in self.nodes:
                    self.nodes[dst] = []
                self.nodes[src].append(dst)

    def is_valid(self, state: set):
        for atom in self.nodes:
            self.visited[atom] = False
        dfs_src = None
        at_robot = None
        for atom in state:
            if "visited(" in atom:
                atom = atom.split("visited(")[1].split(")")[0]
                self.visited[atom] = True
                if dfs_src is None:
                    dfs_src = atom
            if "at-robot(" in atom:
                if at_robot is not None:
                    return False
                at_robot = atom.split("at-robot(")[1].split(")")[0]
        if at_robot is not None and not self.visited[at_robot]:
            return False
        self.dfs(dfs_src)
        for atom in self.visited:
            if self.visited[atom]:
                return False
        return True

    def dfs(self, v):
        if self.visited[v]:
            self.visited[v] = False
            for w in self.nodes[v]:
                self.dfs(w)
