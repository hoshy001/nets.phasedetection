import argparse
import MDAnalysis as mda
import math
import numpy as np
import os
import time
import tqdm
from typing import List, Optional, Tuple


class Pdb2Pts(object):
    def __init__(self,
                 category: str,
                 trans: bool,
                 rot: bool,
                 model: Optional[str] = None,
                 radius: Optional[float] = None,
                 random_centroid: bool = True):
        self.category: str = category
        self.trans: bool = trans
        self.rot: bool = rot
        self.model: Optional[str] = model
        assert self.model in ('Sphere', 'Cube')
        self.radius: Optional[float] = radius
        self.random_centroid: bool = random_centroid

        self.path: str = os.path.join('point_clouds', self.category)

        if not os.path.exists(os.path.join(self.path, 'points')):
            os.makedirs(os.path.join(self.path, 'points'))

        if not os.path.exists(os.path.join(self.path, 'xyz')):
            os.makedirs(os.path.join(self.path, 'xyz'))

    def get_replicate_increment(self,
                                Lx: float,
                                Ly: float,
                                Lz: float,
                                n: int = 3):
        """Return the increment in each direction for replicating the box."""
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

    def replicate_box(self,
                      increment_x: List[int],
                      increment_y: List[int],
                      increment_z: List[int]) -> List[List[int]]:
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

    def replicate_points(self, pos: np.ndarray, replicate_box_vec: List[np.ndarray]) -> np.ndarray:
        """Replicate points in the box."""
        orig_pos = pos
        num_points = orig_pos.shape[0]
        total_num_points = num_points * len(replicate_box_vec)
        pos = np.empty((total_num_points, 3), dtype=orig_pos.dtype)
        n = 0
        for vec in replicate_box_vec:
            pos[n:n + num_points] = orig_pos + vec
            n += num_points
        return pos

    def write_output(self, idx: int, log_file, pos: np.ndarray):
        file_name = f'coord_O_{self.category}_{idx}'

        if log_file:
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

    def rand_rotation_matrix(self) -> np.ndarray:
        """
        Creates a random uniform rotation matrix.
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        theta, phi, z = np.random.uniform(size=(3,))

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

    def apply_periodic_to_pos(self,
                              pos: np.ndarray,
                              Lx: float,
                              Ly: float,
                              Lz: float):
        """Apply periodic boundary conditions."""
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
                              replicate_box_vec: List[np.ndarray],
                              model: Optional[str] = None,
                              radius: Optional[float] = None) -> np.ndarray:
        """"Apply random rotation."""
        # Replicate box so that the transformed coordinates
        # can be wrapped into the original bounding box
        pos = self.replicate_points(pos, replicate_box_vec)

        M = self.rand_rotation_matrix()

        if radius and model == 'Sphere':
            pos = self.select_points_in_sphere_from_box(
                pos, Lx, Ly, Lz, radius)

            pos = np.dot(pos, M)

            centroid = pos.mean(axis=0) - radius
            pos -= centroid
        else:
            pos = np.dot(pos, M)

            if radius:
                pos = self.select_points_in_cube_from_box(
                    pos, Lx, Ly, Lz, radius)

                centroid = pos.mean(axis=0) - radius / 2.0
                pos -= centroid
            else:
                pos = pos[np.logical_and(pos[:, 0] < Lx, pos[:, 0] >= 0)]
                pos = pos[np.logical_and(pos[:, 1] < Ly, pos[:, 1] >= 0)]
                pos = pos[np.logical_and(pos[:, 2] < Lz, pos[:, 2] >= 0)]

        return pos

    def rand_periodic_translation(self, pos: np.ndarray, Lx: float, Ly: float, Lz: float) -> np.ndarray:
        """Randomly moving the atoms along the X, Y, and Z directions
        and apply the periodic boundary conditions on the new positions.
        """
        vec_trans = np.array([np.random.uniform(0, Lx * 0.5),
                              np.random.uniform(0, Ly * 0.5),
                              np.random.uniform(0, Lz * 0.5)])

        new_pos = pos + vec_trans
        new_pos = self.apply_periodic_to_pos(new_pos, Lx, Ly, Lz)

        return new_pos

    def get_points_in_cube(self, pos: np.ndarray, length: float, centroid: np.ndarray) -> np.ndarray:
        """Get the points in the cube of given length and centroid."""
        min_l = centroid - length / 2.0
        max_l = centroid + length / 2.0
        pos = pos[np.logical_and(pos[:, 0] < max_l[0], pos[:, 0] >= min_l[0])]
        pos = pos[np.logical_and(pos[:, 1] < max_l[1], pos[:, 1] >= min_l[1])]
        pos = pos[np.logical_and(pos[:, 2] < max_l[2], pos[:, 2] >= min_l[2])]
        return pos

    def get_points_in_sphere(self, pos: np.ndarray, radius: float, centroid: np.ndarray) -> np.ndarray:
        """Get the points in the sphere of given radius and centroid."""
        diff = np.subtract(pos, centroid)
        dist = np.sum(np.power(diff, 2), axis=1)
        in_sphere = np.where(dist < radius * radius)
        return pos[in_sphere]

    def select_points_in_sphere_from_box(self,
                                         pos: np.ndarray,
                                         Lx: float,
                                         Ly: float,
                                         Lz: float,
                                         radius: float) -> np.ndarray:
        """Select points in sphere using a fixed centroid in the box."""
        if self.random_centroid:
            centroid = np.array([np.random.uniform(0.3 * Lx, 0.7 * Lx),
                                 np.random.uniform(0.3 * Ly, 0.7 * Ly),
                                 np.random.uniform(0.3 * Lz, 0.7 * Lz)])
        else:
            centroid = np.array([Lx / 2.0, Ly / 2.0, Lz / 2.0])

        pos = self.get_points_in_cube(pos, 2.1 * radius, centroid)

        pos = self.get_points_in_sphere(pos, radius, centroid)

        centroid = pos.mean(axis=0)
        pos -= centroid

        return pos

    def select_points_in_cube_from_box(self,
                                       pos: np.ndarray,
                                       Lx: float,
                                       Ly: float,
                                       Lz: float,
                                       length: float) -> np.ndarray:
        """Select points in sphere using a fixed centroid in the box."""
        if self.random_centroid:
            centroid = np.array([np.random.uniform(0.3 * Lx, 0.7 * Lx),
                                 np.random.uniform(0.3 * Ly, 0.7 * Ly),
                                 np.random.uniform(0.3 * Lz, 0.7 * Lz)])
        else:
            centroid = np.array([Lx / 2.0, Ly / 2.0, Lz / 2.0])

        pos = self.get_points_in_cube(pos, length, centroid)

        centroid = pos.mean(axis=0)
        pos -= centroid

        return pos

    def gen_pts_from_pdb_one_worker(self, args: Tuple[str, int, int, str]):
        pdb_file: str = args[0]
        naug: int = args[1]
        idx: int = args[2]
        log_file = args[3]

        universe = mda.Universe(pdb_file)

        Lx, Ly, Lz = universe.dimensions[:3]

        oxygen = universe.select_atoms('name O*')

        if self.radius or self.rot:
            n = 1

            if self.radius:
                min_L = min(Lx, min(Ly, Lz))
                cst = 2.0 if self.model == 'Sphere' else 1.0
                if self.radius * cst > min_L:
                    n = max(n, math.ceil(self.radius * cst / min_L))

            if self.rot:
                n = max(n, 3)

            increment = self.get_replicate_increment(Lx, Ly, Lz, n=n)

            replicate_box_vec = self.replicate_box(*increment)

            for j in range(len(replicate_box_vec)):
                vec = replicate_box_vec[j]
                new_vec = np.multiply([Lx, Ly, Lz], np.array(vec))
                replicate_box_vec[j] = new_vec

        orig_pos_oxygen = oxygen.positions

        # Make sure all atoms are within the periodic box
        orig_pos_oxygen = self.apply_periodic_to_pos(
            orig_pos_oxygen, Lx, Ly, Lz)

        if log_file:
            log_file.write(f'{pdb_file}, N={orig_pos_oxygen.shape[0]}, '
                           f'Lx={Lx}, Ly={Ly}, Lz={Lz}\n')

        idx = idx * naug + 1
        for j in range(naug):
            pos_oxygen = orig_pos_oxygen

            if self.trans:
                # Apply random data augmentation translation
                pos_oxygen = self.rand_periodic_translation(
                    pos_oxygen, Lx, Ly, Lz)

            if self.rot:
                # Apply random data augmentation rotation
                pos_oxygen = self.apply_rotation_to_pos(
                    pos_oxygen, Lx, Ly, Lz, replicate_box_vec,
                    model=self.model, radius=self.radius)

            elif self.radius:
                # Select points within a spherical space in the box
                pos_oxygen = self.replicate_points(
                    pos_oxygen, replicate_box_vec)

                if self.model == 'Sphere':
                    pos_oxygen = self.select_points_in_sphere_from_box(
                        pos_oxygen, Lx, Ly, Lz, self.radius)

                    pos_oxygen += self.radius
                else:
                    pos_oxygen = self.select_points_in_cube_from_box(
                        pos_oxygen, Lx, Ly, Lz, self.radius)

                    pos_oxygen += self.radius / 2.0

            self.write_output(idx, log_file, pos_oxygen)
            idx += 1

        return

    def gen_pts_from_pdb(self, pdb_path: str, naug: int, num_workers: int = 1):
        """Generate point clouds from PDB files."""
        assert naug >= 1
        assert num_workers >= 1

        file_list = os.listdir(pdb_path)
        pdb_files = [i for i in file_list if '.pdb' in i]

        assert len(pdb_files) >= 1, f'there is no pdb file in {pdb_path}'

        if num_workers == 1:
            # Create a log file for later debugging
            with open(f'{self.category}.log', 'w') as log_file:
                for i in tqdm(range(len(pdb_files))):
                    self.gen_pts_from_pdb_one_worker(
                        (os.path.join(pdb_path, pdb_files[i]), naug, i, log_file))

        else:
            from tqdm.contrib.concurrent import process_map

            args = []
            for i in range(len(pdb_files)):
                args.append(
                    (os.path.join(pdb_path, pdb_files[i]), naug, i, None))

            process_map(
                self.gen_pts_from_pdb_one_worker,
                args, max_workers=num_workers)

        return


if __name__ == "__main__":
    phases = [
        'lam (lamellar)',
        'hpc (hexagonally packed cylinder)',
        'hpl (hexagonally perforated lamellar)',
        'bcc (body-centered cubic)',
        'dis (disordered states)',
        'sg (single gyroid)',
        'dg (double gyroid)',
        'dd (double diamond)',
        'p (plumber\'s nightmare)',
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, required=True,
                        help='Category (phase) name: ' + ', '.join(phases))
    parser.add_argument('-t', '--rand_trans', action='store_true',
                        help='Whether to apply random periodic translation')
    parser.add_argument('-r', '--rand_rot', action='store_true',
                        help='Whether to apply random periodic rotation')
    parser.add_argument('-m', '--model', type=str, choices=('Sphere', 'Cube'),
                        help='Model name `Sphere`, or `Cube` to select points from the given volume')
    parser.add_argument('-v', '--volume', type=float,
                        help='Select points within a cubical or spherical volume (Angstrom^3) from point cloud')
    parser.add_argument('-n', '--naug', type=int, default=1,
                        help='Number of data augmentation (translation + rotation) for each point cloud')
    parser.add_argument('-w', '--num_workers', type=int, default=1,
                        help='Number of workers to spawn the pdb processing')

    opt = parser.parse_args()

    opt.category = opt.category.lower()

    pdb_path = os.path.join('raw', 'pdb', opt.category)

    assert os.path.exists(pdb_path), f'Path {pdb_path} does not exist!'

    print(f'Processing pdb files of {opt.category} structure...')

    if opt.rand_trans:
        print(f'Apply random periodic translation')

    if opt.rand_rot:
        print(f'Apply random periodic rotation')

    if opt.volume:
        assert opt.volume > 0.0
        model = opt.model if opt.model else 'Sphere'
        if model == 'Sphere':
            radius = np.ceil(np.power(opt.volume / (4.0 / 3.0 * np.pi), 1/3.))
            print(f'select points within a spherical volume: {opt.volume} (Angstrom^3)'
                  f', with radius of {radius} (Angstrom) from the whole point cloud')
        else:
            radius = np.ceil(np.power(opt.volume, 1/3.))
            print(f'select points within a cubical volume: {opt.volume} (Angstrom^3)'
                  f', with length of {radius} (Angstrom) from the whole point cloud')
    else:
        model = None
        radius = None

    tic = time.perf_counter()
    pdb2pts = Pdb2Pts(opt.category,
                      opt.rand_trans,
                      opt.rand_rot,
                      model=model,
                      radius=radius,
                      random_centroid=True)
    pdb2pts.gen_pts_from_pdb(pdb_path, opt.naug, opt.num_workers)
    toc = time.perf_counter()
    print(
        f'Data augmentation took {int((toc - tic) * 1000)} ms using {opt.num_workers} workers.')
