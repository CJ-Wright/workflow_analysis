__author__ = 'christopher'
from ase.io.trajectory import PickleTrajectory
from simdb.search import *
from simdb.insert import *
from simdb.handlers import FileLocation
from filestore.retrieve import handler_context
import filestore.commands as fsc


def run_simulation(sim):
    # Load info from simulation request
    sim_params, = find_simulation_parameter_document(_id=sim.params.id)

    # TODO: Throw in some statements about timeouts, acceptable U(q), etc.
    iterations = sim_params.iterations
    target_acceptance = sim_params.target_acceptance
    ensemble_temp = sim_params.temperature

    # Find Starting Atoms
    traj_entry, = find_atomic_config_document(_id=sim.atoms.id)
    traj = traj_entry.file_payload
    # Load the atoms
    # We want to continue this simulation
    if isinstance(traj, list):
        # Give back the final configuration
        atoms = traj[-1]
        # Search filestore and get the file_location
        with handler_context({'ase': FileLocation}):
            atoms_file_location = fsc.retrieve(traj_entry.file_uid)
        wtraj = PickleTrajectory(atoms_file_location, 'a')

    # This is a new sim with a new trajectory
    else:
        # Give back the initial config
        atoms = traj
        # Generate new file location and save it to filestore
        new_atoms_entry = insert_atom_document(
            traj_entry.name + '_' + sim.name, atoms)

        with handler_context({'ase': FileLocation}):
            new_file_location = fsc.retrieve(new_atoms_entry.file_uid)
        wtraj = PickleTrajectory(new_file_location, 'w')
        sim.simulation_atoms = new_atoms_entry
        sim.save()

    # Create Calculators
    pes, = find_pes_document(_id=sim.pes.id)
    master_calc = pes.payload

    # Attach MultiCalc to atoms
    atoms.set_calculator(master_calc)

    sim.start_total_energy.append(atoms.get_total_energy())
    sim.start_potential_energy.append(atoms.get_potential_energy())
    sim.start_kinetic_energy.append(atoms.get_kinetic_energy())
    sim.start_time.append(ttime.time())

    # Build Ensemble
    ensemble, = find_ensemble_document(_id=sim.meta_ensemble.id)
    dyn = ensemble.payload(atoms, **ensemble.ensemble_kwargs)

    sim.ran = True
    sim.save()

    # Simulate
    # TODO: eventually support different simulation engines
    out_traj, metadata = dyn.run(iterations)
    sim.end_time.append(ttime.time())

    sim.total_iterations.append(sim.params.iterations)
    sim.total_samples.append(samples)
    sim.leapfrog_per_iter.append(l_p_i)
    sim.finished = True
    sim.seed.append(seed)
    sim.save()
    # Write info to DB
    sim.final_potential_energy.append(out_traj[-1].get_potential_energy())
    sim.final_kinetic_energy.append(out_traj[-1].get_kinetic_energy())
    sim.final_kinetic_energy.append(out_traj[-1].get_total_energy())
    sim.save()
