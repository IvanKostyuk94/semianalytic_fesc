import pandas as pd
import numpy as np
import os
import illustris_python as il
import astropy.units as u
from pyTNG.cosmology import TNGcosmo
from utils import get_sim
import pyTNG.utils as utils
from utils import get_particle_dist
from utils import get_redshift
from sklearn.decomposition import PCA


h = TNGcosmo.h


def get_normed_coord(particles, df, index, z):
    gal_center = np.array(
        [
            df.loc[index]["Halo_pos_x"],
            df.loc[index]["Halo_pos_y"],
            df.loc[index]["Halo_pos_z"],
        ]
    )
    rel_pos = particles["Coordinates"] - gal_center
    dist_to_cm = (1 * u.kpc).to(u.cm).value / h / (1 + z)
    radius = df.loc[index]["r"] * 2 / dist_to_cm
    rel_pos_norm = rel_pos / radius
    particles["rel_pos_norm"] = rel_pos_norm
    return


def map_to_new_dict(particles, relevant):
    rel_particles = {}
    newcount_particles = (relevant == True).sum()
    for key, value in particles.items():
        try:
            rel_particles[key] = value[relevant]
        # for Python scalars
        except TypeError as e:
            if "not subscriptable" in str(e):
                pass
            else:
                raise
        # for numpy scalars
        except IndexError as e:
            if "invalid index to scalar variable" in str(e):
                pass
            else:
                raise
    if "count" in particles:
        rel_particles["count"] = newcount_particles
    return rel_particles


def select_box_particles(particles):
    if particles["count"] == 0:
        return particles
    else:
        idces_rel_particles = ~np.any(np.abs(particles["rel_pos_norm"]) > 1, axis=1)
        rel_particles = map_to_new_dict(particles, idces_rel_particles)

    return rel_particles


def select_sphere_particles(particles, df, idx, z):
    if particles["count"] == 0:
        return particles
    else:
        get_particle_dist(particles, df, idx, z)
        idces_rel_particles = particles["rel_dist"] < 1
        rel_particles = map_to_new_dict(particles, idces_rel_particles)
    return rel_particles


def apply_pca(particles):
    pca = PCA(3)
    pca.fit(particles["Coordinates"])
    particles["TransformedCoordinates"] = pca.transform[particles["Coordinates"]]
    return
