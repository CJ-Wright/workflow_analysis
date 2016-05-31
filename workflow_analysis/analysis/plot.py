from __future__ import print_function
import os
from inspect import isgenerator
from copy import deepcopy as dc

import ase.io as aseio
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
import matplotlib

# from workflow.workflow import sim_unpack
from pyiid.calc import wrap_rw
from pyiid.utils import (tag_surface_atoms,
                         get_angle_list, get_coord_list, get_bond_dist_list
    )
from pyiid.kernels.cpu_nxn import get_d_array, get_r_array
from matplotlib.patches import Rectangle
from decimal import Decimal

__author__ = 'christopher'

font = {'family': 'normal',
        # 'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)
matplotlib.rc('figure', figsize=(8, 6), dpi=80 * 3)
# plt.ion()
colors = ['grey', 'red', 'royalblue']
light_colors = ['silver', 'mistyrose', 'powderblue']


def plot_pdf(r, gobs, gcalc=None, save_file=None, show=True, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if gcalc is not None:
        rw, scale = wrap_rw(gcalc, gobs)
        print('Rw', rw * 100, '%')
        rw = float(Decimal(np.round(rw, 4)).quantize(Decimal('1.0000')))


        gdiff = gobs - gcalc * scale
        baseline = -1 * np.abs(1.5 * gobs.min())
        if np.any(gobs - gdiff < 1):
            baseline -= .5

    if gcalc is not None:
        p0 = ax.plot(0, 0, '-', color='none', label=r"$Rw={}\%$".format(rw * 100.))
        p1 = ax.plot(r, gobs, 'bo', label=r"$G(r)$ target")
        p2 = ax.plot(r, gcalc * scale, 'r-', label=r"$G(r)$ fit")
        p3 = ax.plot(r, gdiff + baseline, 'g-', label=r"$G(r)$ diff")
        ax.axhline(y=baseline, color='k', linestyle=':')
        ax.set_ylim(np.min(gdiff + baseline) * 1.2, np.max(gobs) * 1.5)
        ax.set_xlim(np.min(r), np.max(r))
    elif isinstance(gobs, list) and 'labels' in kwargs:
        for g, l in zip(gobs, kwargs['labels']):
            p1 = ax.plot(r, g, '-', label=l)
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel(r"$G (\AA^{-2})$")
    plt.legend(loc='best', prop={'size': 18}, fancybox=True, framealpha=0.3)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file + '_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_pdf.png', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_pdf.pdf', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_waterfall_pdf(r, gcalcs, gobs, save_file=None, show=True,
                       **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(r, gobs, 'bo', label="G(r) data")
    for i, gcalc in enumerate(gcalcs):
        rw, scale = wrap_rw(gcalc, gobs)
        print(i, 'Rw', rw * 100, '%')
        plt.plot(r, gcalc * scale + i, '-', label="Fit {}".format(i))
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel(r"$G (\AA^{-2})$")
    ax.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_pdf.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_waterfall_diff_pdf(r, gcalcs, gobs, save_file=None, show=True,
                            **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, gcalc in enumerate(gcalcs):
        rw, scale = wrap_rw(gcalc, gobs)
        print(i, 'Rw', rw * 100, '%')
        plt.plot(r, gobs - (gcalc * scale)
                 # - i
                 , '-', label="Fit {}".format(i))
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel(r"$G (\AA^{-2})$")
    ax.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_pdf.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_waterfall_pdf_2d(r, gcalcs, gobs, save_file=None, show=True,
                          **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(r, gobs, 'bo', label="G(r) data")
    gcalcs_img = []
    for i, gcalc in enumerate(gcalcs):
        rw, scale = wrap_rw(gcalc, gobs)
        print(i, 'Rw', rw * 100, '%')
        gcalcs.append(gcalc * scale)
    ax.imshow(gcalcs_img, aspect='auto', origin='lower',
              extent=(r.min(), r.max(), 0, len(gcalcs_img)))
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel("iteration")
    ax.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_2d_water_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_2d_water_pdf.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_waterfall_diff_pdf_2d(r, gcalcs, gobs, save_file=None, show=True,
                               **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(r, gobs, 'bo', label="G(r) data")
    gcalcs_img = []
    for i, gcalc in enumerate(gcalcs):
        rw, scale = wrap_rw(gcalc, gobs)
        print(i, 'Rw', rw * 100, '%')
        gcalcs.append(gobs - gcalc * scale)
    ax.imshow(gcalcs_img, aspect='auto', origin='lower',
              extent=(r.min(), r.max(), 0, len(gcalcs_img)))
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel("iteration")
    ax.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_2d_water_diff_pdf.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_2d_water_diff_pdf.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_angle(cut, start, finish, target=None, save_file=None, show=True,
               **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stru_l = {}

    # If the PDF document created with atomic config, use that as target
    if target is not None:
        stru_l['Target'] = target
    stru_l['Start'] = start
    stru_l['Finish'] = finish
    for atoms in stru_l.values():
        if len(set(atoms.get_tags())) == 1:
            tag_surface_atoms(atoms)

    symbols = set(stru_l['Start'].get_chemical_symbols())

    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    if len(tags) == 1:
        tags = {'': (1, '*')}
    # need to change this
    bins = np.linspace(0, 180, 100)
    # Bin the data
    for n, key in enumerate(stru_l.keys()):
        for symbol in symbols:
            for tag in tags.keys():
                a, b = np.histogram(
                    get_angle_list(stru_l[key], cut, element=symbol,
                                   tag=tags[tag][0]), bins=bins)
                if False:
                    pass
                    # if np.alltrue(stru_l[key].pbc):
                    # crystal
                    # for y, x in zip(a, b[:-1]):
                    #     plt.axvline(x=x, ymax=y, color='grey', linestyle='--')
                else:
                    total = np.sum(a)
                    ax.plot(b[:-1], a,
                            label='{0} {1} {2}, {3}'.format(key,
                                                            symbol,
                                                            tag,
                                                            total),
                            marker=tags[tag][1], color=colors[n])
    ax.set_xlabel('Bond angle in Degrees')
    ax.set_xlim(0, 180)
    ax.set_ylabel('Angle Counts')
    ax.legend(loc='best', prop={'size': 18}, fancybox=True, framealpha=0.3)
    if save_file is not None:
        plt.savefig(save_file + '_angle.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.png', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.pdf', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()


def plot_core_shell_angle(cut, start, finish, target=None, save_file=None,
                          show=True, **kwargs):
    fig, axes = plt.subplots(2, sharey=True)
    stru_l = {}

    # If the PDF document created with atomic config, use that as target
    if target is not None:
        stru_l['Target'] = target
    stru_l['Start'] = start
    stru_l['Finish'] = finish
    for atoms in stru_l.values():
        if len(set(atoms.get_tags())) == 1:
            tag_surface_atoms(atoms)

    symbols = set(stru_l['Start'].get_chemical_symbols())

    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    if len(tags) == 1:
        tags = {'': (1, '*')}
    # need to change this
    bins = np.linspace(0, 180, 100)
    # Bin the data
    for n, key in enumerate(stru_l.keys()):
        for symbol in symbols:
            for ax, tag in zip(axes, tags.keys()):
                a, b = np.histogram(
                    get_angle_list(stru_l[key], cut, element=symbol,
                                   tag=tags[tag][0]), bins=bins)
                total = np.sum(a)
                if len(symbols) > 1:
                    ax.plot(
                        b[:-1], a,
                        # label='{0} {1} {2}, {3}'.format(key, symbol, tag, total),
                        label='{0} {1}'.format(key, symbol),
                        marker='o', color=colors[n])
                else:
                    ax.plot(
                        b[:-1], a,
                        # label='{0} {1} {2}, {3}'.format(key, symbol, tag, total),
                        label='{0}'.format(key),
                        marker='o', color=colors[n])
    for i, ax in enumerate(axes):
        if i == 1:
            ax.set_xlabel('Bond angle in Degrees')
        ax.set_xlim(0, 180)
        ax.set_ylabel('Angle Counts')
        ax.legend(loc='best', prop={'size': 18}, fancybox=True, framealpha=0.3)
    if save_file is not None:
        plt.savefig(save_file + '_angle.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.png', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.pdf', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()


def plot_coordination(cut, start, finish, target=None,
                      save_file=None,
                      show=True, index=-1, **kwargs):
    stru_l = {}
    # If the PDF document created with atomic config, use that as target
    if target is not None:
        stru_l['Target'] = target
    stru_l['Start'] = start
    stru_l['Finish'] = finish
    for atoms in stru_l.values():
        if len(set(atoms.get_tags())) == 1:
            tag_surface_atoms(atoms)

    symbols = set(stru_l.itervalues().next().get_chemical_symbols())
    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l.itervalues().next()[
            [atom.index for atom in stru_l.itervalues().next() if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    if len(tags) == 1:
        tags = {'': (1, '*')}
    b_min = None
    b_max = None
    for key in stru_l.keys():
        total_coordination = get_coord_list(stru_l[key], cut)
        l_min = min(total_coordination)
        l_max = max(total_coordination)
        if b_min is None or b_min > l_min:
            b_min = l_min
        if b_max is None or b_max < l_max:
            b_max = l_max
    if b_min == b_max:
        bins = np.asarray([b_min, b_max])
    else:
        bins = np.arange(b_min, b_max + 2)
    width = 3. / 4 / len(stru_l)
    offset = .3 * 3 / len(stru_l)
    patterns = ('x', '\\', 'o', '.', '\\', '*')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n, key in enumerate(stru_l.keys()):
        bottoms = np.zeros(bins.shape)
        j = 0
        for symbol in symbols:
            for tag, use_colors in zip(tags.keys(), [colors, light_colors]):
                hatch = patterns[j]
                coord = get_coord_list(stru_l[key], cut, element=symbol,
                                       tag=tags[tag][0])
                a, b = np.histogram(coord, bins=bins)
                total = np.sum(a)
                if len(symbols) > 1:
                    ax.bar(b[:-1] + n * offset, a, width, bottom=bottoms[:-1],
                           color=use_colors[n],
                           label='{0} {1} {2}, {3}'.format(key, symbol, tag,
                                                           total),
                           )
                else:
                    ax.bar(b[:-1] + n * offset, a, width, bottom=bottoms[:-1],
                           color=use_colors[n],
                           label='{} {}, {}'.format(key, tag, total),
                           )
                j += 1
                bottoms[:-1] += a

    ax.set_xlabel('Coordination Number')
    ax.set_xticks(bins[:-1] + 1 / 2.)
    ax.set_xticklabels(bins[:-1])
    ax.set_ylabel('Atomic Counts')
    ax2 = plt.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax.legend(loc='best', prop={'size': 18}, fancybox=True, framealpha=0.3)
    if save_file is not None:
        for ext in ['eps', 'png', 'pdf']:
            plt.savefig(save_file + '_coord.' + ext, bbox_inches='tight',
                        transparent='True')
    if show is True:
        plt.show()
    return


def plot_radial_bond_length(cut, start, finish, target=None,
                            save_file=None,
                            show=True, index=-1, **kwargs):
    stru_l = {}
    # If the PDF document created with atomic config, use that as target
    if target is not None:
        stru_l['Target'] = target
    stru_l['Start'] = start
    stru_l['Finish'] = finish

    # Make subplots for each structure
    fig, axes = plt.subplots(len(stru_l.keys()), sharey=True)
    maxdist = 0.
    maxbond = 0.
    minbond = 10.

    # For each axis/structure pair
    for n, (key, ax) in enumerate(zip(stru_l.keys(), axes)):
        atoms = stru_l[key]
        com = atoms.get_center_of_mass()
        dist_from_center = []
        bond_lengths = []
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                dist = atoms.get_distance(i, j)
                if dist <= cut and dist != 0.0:
                    ave_position = (atoms[i].position + atoms[j].position)/2.
                    dist_from_center.append(np.sqrt(np.sum((ave_position - com)**2)))
                    bond_lengths.append(atoms.get_distance(i, j))
        ax.scatter(dist_from_center, bond_lengths, c=colors[n], marker='o',
                   label='{0}'.format(key), s=40)
        if np.max(dist_from_center) > maxdist:
            maxdist = np.max(dist_from_center)
        if np.max(bond_lengths) > maxbond:
            maxbond = np.max(bond_lengths)
        if np.min(bond_lengths) < minbond:
            minbond = np.min(bond_lengths)

            # ax.legend(loc='best', prop={'size': 12})

    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, maxdist + .5)
        ax.set_ylim(minbond - .1, maxbond + .1)
        plt.locator_params(axis='y', nbins=8)
        ax2 = ax.twinx()
        ax2.set_ylabel(stru_l.keys()[i])
        ax2.set_yticks([])
        if i == 1:
            ax.set_ylabel('Bond Distance $(\AA)$')
    ax.set_xlabel('Distance from Center $(\AA)$')

    if save_file is not None:
        plt.savefig(save_file + '_rbonds.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_rbonds.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return


def plot_bonds(sim, cut, save_file=None, show=True, index=-1):
    atomic_config, = find_atomic_config_document(_id=sim.atoms.id)
    traj = atomic_config.file_payload
    stru_l = {}
    # If the PDF document created with atomic config, use that as target
    cl = sim.pes.calc_list
    for calc in cl:
        if calc.calculator == 'PDF':
            break
    # If we used a theoretical target structure, get it and name it
    # if calc.ase_config_id is not None:
    #     target_atoms, = find_atomic_config_document(_id=calc.ase_config_id)
    #     stru_l['Target'] = target_atoms.file_payload

    stru_l['Start'] = traj[0]
    stru_l['Finish'] = traj[index]
    for atoms in stru_l.values():
        tag_surface_atoms(atoms, cut)

    symbols = set(stru_l['Start'].get_chemical_symbols())
    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l['Start'][
            [atom.index for atom in stru_l['Start'] if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    linestyles = ['-', '--', ':']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n, key in enumerate(stru_l.keys()):
        for k, symbol in enumerate(symbols):
            for tag in tags.keys():
                bonds = get_bond_dist_list(
                    stru_l[key], cut, element=symbol, tag=tags[tag][0])
                a, b = np.histogram(bonds, bins=10)
                plt.plot(b[:-1], a, linestyles[k],
                         label=key + ' ' + symbol + ' ' + tag,
                         marker=tags[tag][1], color=colors[n])
    ax.set_xlabel('Bond distance in angstrom')
    ax.set_ylabel('Bond Counts')
    plt.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_angle.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_angle.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()


def plot_average_coordination(cut, start, finish, target=None,
                              save_file=None,
                              show=True, index=-1, **kwargs):
    stru_l = {}
    # If the PDF document created with atomic config, use that as target
    if target is not None:
        stru_l['Target'] = target
    stru_l['Start'] = start
    stru_l['Equilibrium'] = finish
    for atoms in stru_l.values():
        tag_surface_atoms(atoms)

    symbols = set(stru_l.itervalues().next().get_chemical_symbols())
    tags = {'Core': (0, '+'), 'Surface': (1, '*')}
    for tag in tags.keys():
        tagged_atoms = stru_l.itervalues().next()[
            [atom.index for atom in stru_l.itervalues().next() if
             atom.tag == tags[tag][0]]]
        if len(tagged_atoms) == 0:
            del tags[tag]
    if len(tags) == 1:
        tags = {'': (1, '*')}
    b_min = None
    b_max = None
    for key in stru_l.keys():
        total_coordination = get_coord_list(stru_l[key], cut)
        l_min = min(total_coordination)
        l_max = max(total_coordination)
        if b_min is None or b_min > l_min:
            b_min = l_min
        if b_max is None or b_max < l_max:
            b_max = l_max
    if b_min == b_max:
        bins = np.asarray([b_min, b_max])
    else:
        bins = np.arange(b_min, b_max + 2)
    width = 3. / 4 / len(stru_l)
    offset = .3 * 3 / len(stru_l)
    patterns = ('x', '\\', 'o', '.', '\\', '*')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n, key in enumerate(stru_l.keys()):
        bottoms = np.zeros(bins.shape)
        j = 0
        for symbol in symbols:
            for tag in tags.keys():
                hatch = patterns[j]
                coord = get_coord_list(stru_l[key], cut, element=symbol,
                                       tag=tags[tag][0])
                a, b = np.histogram(coord, bins=bins)
                total = np.sum(a)
                ax.bar(b[:-1] + n * offset, a, width, bottom=bottoms[:-1],
                       color=colors[n],
                       label='{0} {1} {2}, {3}'.format(key, symbol, tag,
                                                       total),
                       hatch=hatch)
                j += 1
                bottoms[:-1] += a

    ax.set_xlabel('Coordination Number')
    ax.set_xticks(bins[:-1] + 1 / 2.)
    ax.set_xticklabels(bins[:-1])
    ax.set_ylabel('Atomic Counts')
    ax2 = plt.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax.legend(loc='best', prop={'size': 12})
    if save_file is not None:
        plt.savefig(save_file + '_coord.eps', bbox_inches='tight',
                    transparent='True')
        plt.savefig(save_file + '_coord.png', bbox_inches='tight',
                    transparent='True')
    if show is True:
        plt.show()
    return

def plot_bond_structure(atoms, cut):
    pass




if __name__ == '__main__':
    from pyiid.experiments.elasticscatter import ElasticScatter

    src = '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/oldversion.d/misc_figures'
    dest = '/mnt/bulk-data/Dropbox/BNL_Project/HMC_paper/new_figures'
    dirs = [os.path.join(src, f) for f in os.listdir(src) if
            os.path.isdir(os.path.join(src, f))]

    s = ElasticScatter({'rmax': 16, 'rmin':1.5})
    cut = 3.5
    for d in dirs:
        print(d)
        files = os.listdir(d)
        file_names = []
        for f_stem in ['start', 'target', 'min']:
            for f in files:
                if f.endswith(f_stem + '.xyz') and 'half' not in f:
                    file_names.append(os.path.join(src, d, f))
        if len(file_names) == 3:
            base_fn = os.path.split(file_names[0])[-1]
            s_base_fn = base_fn.split('_')[0]
            base_name = os.path.join(dest, os.path.split(d)[-1], s_base_fn)
            print(base_name)
            structures = [aseio.read(f) for f in file_names]
            start_structure, target_structure, min_structure = structures

            # '''
            plot_pdf(gobs=s.get_pdf(target_structure),
                     gcalc=s.get_pdf(min_structure),
                     r=s.get_r(),
                     show=False,
                     save_file=base_name
                     )
            # '''
            '''
            plot_radial_bond_length(cut, start_structure, min_structure,
                                    target_structure,
                                    # show=False,
                                    # save_file=base_name
                                    )
            # '''
            '''
            plot_coordination(cut, start_structure, min_structure,
                              target_structure,
                              # show=False,
                              # save_file=base_name
                              )
            # '''
            '''
            plot_core_shell_angle(cut, start_structure, min_structure,
                                  target_structure,
                                  # show=False,
                                  # save_file=base_name
                                  )
            '''
