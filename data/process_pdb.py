import argparse
import MDAnalysis as mda
import math
import numpy as np
import os
import time
from tqdm import tqdm


class Pdb2Pts(object):
    def __init__(self, category: str, trans: bool, rot: bool):

        self.category: str = category
        self.trans: bool = trans
        self.rot: bool = rot
        self.path: str = os.path.join('point_clouds', self.category)

        if not os.path.exists(os.path.join(self.path, 'points')):
            os.makedirs(os.path.join(self.path, 'points'))

        if not os.path.exists(os.path.join(self.path, 'xyz')):
            os.makedirs(os.path.join(self.path, 'xyz'))

    def apply_periodic_to_pos(self,
                              pos: np.ndarray,
                              Lx: float,
                              Ly: float,
                              Lz: float):
        """
        Apply periodic boundary conditions
        """
        invLx, invLy, invLz = (1.0/Lx, 1.0/Ly, 1.0/Lz)
        X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
        scale_factor = np.floor(Z * invLz)
        Z -= scale_factor * Lz
        scale_factor = np.floor(Y * invLy)
        Y -= scale_factor * Ly
        scale_factor = np.floor(X * invLx)
        X -= scale_factor * Lx
        return pos

    def apply_rotation_to_pos(self,
                              pos: np.ndarray,
                              Lx: float,
                              Ly: float,
                              Lz: float,
                              replicate_box_vec: list):

        # Replicate box so that the transformed coordinates
        # can be wrapped into the original bounding box
        orig_pos = pos

        num_oxygen = orig_pos.shape[0]
        total_num_oxygen = num_oxygen * len(replicate_box_vec)
        pos = np.empty((total_num_oxygen, 3), dtype=orig_pos.dtype)

        n = 0
        for vec in replicate_box_vec:
            pos[n:n + num_oxygen] = orig_pos + vec
            n += num_oxygen

        M = self.rand_rotation_matrix()
        pos = np.dot(pos, M)
        pos = pos[np.logical_and(pos[:, 0] < Lx, pos[:, 0] >= 0)]
        pos = pos[np.logical_and(pos[:, 1] < Ly, pos[:, 1] >= 0)]
        pos = pos[np.logical_and(pos[:, 2] < Lz, pos[:, 2] >= 0)]
        return pos

    def get_replicate_increment(self,
                                Lx: float,
                                Ly: float,
                                Lz: float,
                                n: int = 3):
        """
        Return the increment in each direction for replicating the box.
        """
        assert n >= 1

        increment_x = [i for i in range(-n, n + 1)]
        increment_y = increment_x
        increment_z = increment_x

        if Lx >= max(Ly, Lz):
            if math.floor(Lx / Ly) > 2:
                m = max(math.floor(Lx / Ly), n)
                increment_y = [i for i in range(-m, m + 1)]

            if math.floor(Lx / Lz) > 2:
                m = max(math.floor(Lx / Lz), n)
                increment_z = [i for i in range(-m, m + 1)]

        elif Ly >= max(Lx, Lz):
            if math.floor(Ly / Lx) > 2:
                m = max(math.floor(Ly / Lx), n)
                increment_x = [i for i in range(-m, m + 1)]

            if math.floor(Ly / Lz) > 2:
                m = max(math.floor(Ly / Lz), n)
                increment_z = [i for i in range(-m, m + 1)]

        elif Lz >= max(Lx, Ly):
            if math.floor(Lz / Lx) > 2:
                m = max(math.floor(Lz / Lx), n)
                increment_x = [i for i in range(-m, m + 1)]

            if math.floor(Lz / Ly) > 2:
                m = max(math.floor(Lz / Ly), n)
                increment_y = [i for i in range(-m, m + 1)]

        return increment_x, increment_y, increment_z

    def replicate_box(self, increment_x: list, increment_y: list, increment_z: list):
        """
        Input an increment array in each direction, e.g.
        increment_x, y, z = [-1, 0, 1], replicating the box three times in
        each dimension
        """
        replicate_box_vec = []
        for i in increment_x:
            for j in increment_y:
                for k in increment_z:
                    replicate_box_vec.append([i, j, k])
        return replicate_box_vec

    def rand_rotation_matrix(self):
        """
        Creates a random uniform rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return M

    def rand_periodic_translation(self, pos: np.ndarray, Lx: float, Ly: float, Lz: float):
        """Randomly moving the atoms along the X, Y, and Z directions
        and apply the periodic boundary conditions on the new positions.
        """
        vec_trans = np.array([np.random.uniform(0, Lx * 0.5),
                              np.random.uniform(0, Ly * 0.5),
                              np.random.uniform(0, Lz * 0.5)])

        new_pos = pos + vec_trans
        new_pos = self.apply_periodic_to_pos(new_pos, Lx, Ly, Lz)

        return new_pos

    def write_output(self, idx: int, log_file, pos: np.ndarray):
        file_name = f'coord_O_{self.category}_{idx}'

        log_file.write(f'{pos.shape[0]} {file_name}\n')

        # Save .pts files
        with open(os.path.join(self.path, 'points', f'{file_name}.pts'), 'w') as pts_file:
            for k in range(pos.shape[0]):
                pts_file.write(
                    f'{pos[k, 0]} {pos[k, 1]} {pos[k, 2]}\n')

        # Save .xyz files
        with open(os.path.join(self.path, 'xyz', f'{file_name}.xyz'), 'w') as xyz_file:
            xyz_file.write(str(pos.shape[0]) + '\n\n')
            for k in range(pos.shape[0]):
                xyz_file.write(
                    f'O  {pos[k, 0]} {pos[k, 1]} {pos[k, 2]}\n')

    # pdb file must contain 1 frame
    def gen_pts_from_pdb(self, pdb_path: str, ntrans: int):
        assert ntrans >= 1
        file_list = os.listdir(pdb_path)
        pdb_files = [i for i in file_list if '.pdb' in i]
        idx = 1
        # Create a log file for later debugging
        log_file = open(f'{self.category}.log', 'w')
        for i in tqdm(range(1, len(pdb_files) + 1)):
            universe = mda.Universe(os.path.join(pdb_path, pdb_files[i - 1]))
            oxygen = universe.select_atoms('name O*')
            Lx, Ly, Lz = universe.dimensions[:3]

            if self.rot:
                increment = self.get_replicate_increment(Lx, Ly, Lz, n=3)
                replicate_box_vec = self.replicate_box(*increment)
                for j in range(len(replicate_box_vec)):
                    vec = replicate_box_vec[j]
                    new_vec = np.multiply([Lx, Ly, Lz], np.array(vec))
                    replicate_box_vec[j] = new_vec

            orig_pos_oxygen = oxygen.positions

            # Make sure all atoms are within the periodic box
            orig_pos_oxygen = self.apply_periodic_to_pos(
                orig_pos_oxygen, Lx, Ly, Lz)

            log_file.write(f'{pdb_files[i - 1]}, N={orig_pos_oxygen.shape[0]}, '
                           f'Lx={Lx}, Ly={Ly}, Lz={Lz}\n')

            for j in range(ntrans):
                pos_oxygen = orig_pos_oxygen

                if self.trans:
                    # Apply random data augmentation translation
                    pos_oxygen = self.rand_periodic_translation(
                        pos_oxygen, Lx, Ly, Lz)

                if self.rot:
                    # Apply random data augmentation rotation
                    pos_oxygen = self.apply_rotation_to_pos(
                        pos_oxygen, Lx, Ly, Lz, replicate_box_vec)

                self.write_output(idx, log_file, pos_oxygen)
                idx += 1

        log_file.close()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, required=True,
                        help='category (phase) name: lam (lamellar), hpc (hexagonally packed cylinder), hpl (hexagonally perforated lamellar), bcc (body-centered cubic), dis (disordered states), sg (single gyroid), dg (double gyroid), dd (double diamond), p (plumber\'s nightmare')
    parser.add_argument('-t', '--rand_trans', action='store_true',
                        help='whether to apply random periodic translation')
    parser.add_argument('-r', '--rand_rot', action='store_true',
                        help='whether to apply random periodic rotation')
    parser.add_argument('-nt',
                        '--ntrans', type=int, default=1, help='number of random data augmentation (translation + rotation) for each point cloud')
    opt = parser.parse_args()
    opt.category = opt.category.lower()

    pdb_path = os.path.join('raw', 'pdb', opt.category)

    print(f'Processing pdb files of {opt.category} structure...')
    print(f'Apply random periodic translation is {bool(opt.rand_trans)}')
    print(f'Apply random periodic rotation is {bool(opt.rand_rot)}')

    tic = time.perf_counter()
    Pdb2Pts(opt.category, opt.rand_trans, opt.rand_rot).gen_pts_from_pdb(
        pdb_path, opt.ntrans)
    toc = time.perf_counter()
    print(f'Used {int((toc - tic) * 1000)} ms.')
