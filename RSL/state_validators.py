from math import sqrt, ceil
import tarski


class state_space_validator:
    def __init__(
        self,
        instance_name: str,
        atoms: tarski.util.SymbolIndex,
        init_atoms: tarski.model.Model,
    ):
        print(instance_name)
        domain_name, instance_name = instance_name.split("/")[-2:]
        state_space_file = f"scripts/state_space/{domain_name}_{instance_name.split('.pddl')[0]}.state_space"
        try:
            with open(
                state_space_file,
            ) as f:
                lines = [l if l[-1] != "\n" else l[:-1] for l in f.readlines()]
        except FileNotFoundError:
            print(f"State space not found: {state_space_file}")
            exit(0)
        assert "Atom" in lines[0]
        self.atoms = [a.split("Atom ")[1].replace(" ", "") for a in lines[0].split(";")]
        self.states = []
        for state in lines[1:]:
            self.states.append(self.converter(state, len(self.atoms)))
        # Assert that atoms of state space are equals to rsl atoms
        atoms = [str(a) for a in atoms]
        for atom in self.atoms:
            assert atom in atoms

    def is_valid(self, state: set):
        # there are rsl atoms that do not exist in fd atoms (i.e. blocks on(a,a)),
        # if it appears in state then state is invalid
        for atom in state:
            if atom not in self.atoms:
                return False
        s = self.state2bin(state)
        for state in self.states:
            if s == state:
                return True
        return False

    def state2bin(self, state: set):
        b = ""
        for atom in self.atoms:
            b += "1" if atom in state else "0"
        return b

    def converter(self, line_state: str, length: int):
        decimals = line_state.split(" ")
        assert ceil(length / 64) == len(decimals)
        binary = ""
        for i in range(len(decimals)):
            b = str(bin(int(decimals[i])))[2:]
            zeros = 64 - len(b)
            if i == 0 and length % 64 > 0:
                zeros = length % 64 - len(b)
                assert zeros >= 0
            binary += ("0" * zeros) + b
        return binary


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
        ontable = []
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
            if "ontable(" in atom:
                v = atom.split("ontable(")[1].split(")")[0]
                ontable.append(v)
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
                if v in ontable:
                    return False
        return True


class npuzzle_state_validator:
    def __init__(self, init_atoms: tarski.model.Model):
        self.init_atoms = init_atoms
        (
            self.npuzzle,
            self.size,
            self.size_side,
            self.max_tile,
        ) = self.fill_npuzzle_matrix()

        self.goal = set() # The first state that appears in the regression is the goal.

    def fill_npuzzle_matrix(self):
        max_tile = 0
        for atom in [str(atom) for atom in self.init_atoms.as_atoms()]:
            atom_split = atom.split("(")
            arg_split = [x.strip(")") for x in atom_split[1].split(",")]
            if atom_split[0] == "at":
                tile = int(arg_split[0][2:])
                if tile > max_tile:
                    max_tile = tile
        board_size = max_tile + 1
        size = int(sqrt(board_size))
        return ([[None] * size for i in range(size)], board_size, size, max_tile)

    def is_valid(self, state):
        # Parses the npuzzle state and checks if all tiles appear once.
        if not self.parse_npuzzle_state(state):
            return False

        # Checks if, for example, there're two tiles in the same place, so in this case there
        # would be a "None" position in the board -- because one that should've been occupied
        # is empty.
        if self.npuzzle_matrix_is_not_valid():
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
        if len(self.goal) != 0 and state != self.goal and len(state) != self.size:
            return False

        tile_appears = [False] * self.size
        for atom in state:
            atom_split = atom.split("(")
            arg_split = [x.strip(")") for x in atom_split[1].split(",")]
            if atom_split[0] == "at":
                # at(t_n, p_x_y)
                coords = arg_split[1].split("_")
                x, y = int(coords[1]), int(coords[2])
                tile = int(arg_split[0][2:])
                self.npuzzle[x - 1][y - 1] = tile
                tile_appears[tile] = True

            elif atom_split[0] == "empty":
                # empty(p_x_y)
                coords = arg_split[0].split("_")
                x, y = int(coords[1]), int(coords[2])
                self.npuzzle[x - 1][y - 1] = 0
                tile_appears[0] = True

        # Special goal case
        if len(self.goal) == 0 or state == self.goal:
            last_idx = int(sqrt(len(state) + 1)) - 1
            self.npuzzle[last_idx][last_idx] = 0
            tile_appears[0] = True
            self.goal = state # Now it'll never enter here again.

        return all(tile_appears)


    def npuzzle_matrix_is_not_valid(self):
        for i in range(len(self.npuzzle)):
            for j in range(len(self.npuzzle)):
                if self.npuzzle[i][j] == None:
                    return True

        return False

    def clear_npuzzle_matrix(self):
        size = int(sqrt(self.size))
        self.npuzzle = [[None for i in range(size)] for j in range(size)]


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
                o, _ = (
                    atom.split("at(" if "at(" in atom else "in(")[1]
                    .split(")")[0]
                    .split(",")
                )
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
