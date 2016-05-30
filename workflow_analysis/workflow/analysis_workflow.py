from workflow_analysis.analysis.plot import *

def save_config(new_dir_path, name, d, index=-1, rotation_atoms=None):
    if rotation_atoms is None:
        atoms = d['traj'][index]
        n = len(atoms)
        q = atoms.positions
        dist = np.zeros((n, n, 3), np.float32)
        r = np.zeros((n, n), np.float32)
        get_d_array(dist, q)
        get_r_array(r, dist)
        maxpos = np.argmax(r)
        rotation_atoms = np.unravel_index(maxpos, r.shape)
    out_l = [
        d['traj'][index],
        d['target'],
        d['traj'][0]
    ]
    append_names = ['_min', '_target', '_start']
    file_endings = ['.eps', '.png', '.xyz', '.pov']

    for atoms, an in zip(out_l, append_names):
        atoms.center()
        # Rotate the config onto the viewing axis
        atoms.rotate(atoms[rotation_atoms[0]].position - atoms[
            rotation_atoms[1]].position, 'z')
        atoms.center()
        # save the total configuration
        for e in file_endings:
            file_name = os.path.join(new_dir_path, name + an + e)
            aseio.write(file_name, atoms)
        # cut the config in half along the xy plane
        atoms2 = dc(atoms)
        atoms2.set_constraint()
        atoms2.translate(-1 * atoms2.get_center_of_mass())
        print(atoms2.positions)
        del atoms2[[atom.index for atom in atoms2 if atom.position[2] >= 0]]
        for e in file_endings:
            file_name = os.path.join(new_dir_path, name + '_half' + an + e)
            aseio.write(file_name, atoms2)


def mass_plot(sims, cut, analysis_type='last'):
    if not isgenerator(sims) and not isinstance(sims, list):
        sims = [sims]
    for sim in sims:
        d = sim_unpack(sim)
        if analysis_type == 'min':
            pel = []
            for atoms in d['traj']:
                if atoms._calc is not None:
                    pel.append(atoms.get_potential_energy())
            index = np.argmin(pel)
            print(index)
            print(pel[index])
        elif analysis_type == 'last':
            index = -1
        ase_view(**d)
        plot_pdf(atoms=d['traj'][index], **d)
        plot_angle(cut, index=index, **d)
        plot_coordination(cut, index=index, **d)
        plot_radial_bond_length(cut, index=index, **d)


def mass_save(sims, cut, destination_dir, analysis_type='last'):
    if not isgenerator(sims) or not isinstance(sims, list):
        sims = [sims]
    for sim in sims:
        name = str(sim.atoms.id)
        new_dir_path = os.path.join(destination_dir, str(sim.name))
        if not os.path.exists(new_dir_path):
            os.mkdir(os.path.join(destination_dir, sim.name))
        d = sim_unpack(sim)
        if analysis_type == 'min':
            pel = []
            for atoms in d['traj']:
                if atoms._calc is not None:
                    pel.append(atoms.get_potential_energy())
            index = int(np.argmin(pel))
            print(index)
            print(pel[index])
        elif analysis_type == 'last':
            index = -1

        save_config(new_dir_path, name, d, index)
        plot_pdf(atoms=d['traj'][index], show=False,
                 save_file=os.path.join(new_dir_path, name), **d)

        plot_angle(cut, show=False, save_file=os.path.join(new_dir_path, name),
                   index=index, **d)

        plot_coordination(cut, show=False,
                          save_file=os.path.join(new_dir_path, name),
                          index=index, **d)

        plot_radial_bond_length(cut, show=False,
                                save_file=os.path.join(new_dir_path, name),
                                index=index, **d)
