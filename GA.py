import os
import gc
import sys
#import yaml
import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    print('Could not load CuPy')
    
from geneticalgorithm import geneticalgorithm as ga
from helpers import modulation_envelope
import time
import scipy
from concurrent.futures import ThreadPoolExecutor
from utils import objective_df_cp
#### Argument parsing
# helps = {
#     'settings-file' : "File having the settings to be loaded",
#     'model' : "Name of the model. Selection from the settings file",
# }

# parser = ArgumentParser(description=__doc__,
#                         formatter_class=RawDescriptionHelpFormatter)
# parser.add_argument('--version', action='version', version='%(prog)s')
# parser.add_argument('--settings-file', metavar='str', type=str,
#                     action='store', dest='settings_file',
#                     default=None, help=helps['settings-file'], required=True)
# parser.add_argument('--npz-file', metavar='str', type=str,
#                     action='store', dest='npz_file',
#                     default='real_brain', help=helps['model'], required=True)
# parser.add_argument('--save-dir', metavar='str', type=str,
#                     action='store', dest='save_dir',
#                     default=None, required=True)
#options = parser.parse_args()
#### Argument parsing

# with open(os.path.realpath(options.settings_file)) as stream:
#     settings = yaml.safe_load(stream)

# if os.name == 'nt':
#     extra_path = '_windows'
# else:
#     extra_path = ''

# sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))



def objective_df(x, field_data, regions_of_interest, aal_regions, region_volumes, currents, threshold, pref_dir = None, cortex_region = 3, mode='free', parallel=False):  
    def processCurrent(current):
        # Compute field for freq 1
        e_field_base = current[0]*field_data[electrodes[0]] - current[0]*field_data[electrodes[1]]
        # Compute field for freq 2
        e_field_df = current[1]*field_data[electrodes[2]] - current[1]*field_data[electrodes[3]]
        # Compyte modulated field
        if mode == 'free':
            modulation_values = modulation_envelope(e_field_base, e_field_df)
        elif mode == 'pref':
            modulation_values = 2 * np.minimum(np.abs(e_field_base), np.abs(e_field_df))
        elif mode == 'comb':
            cortex_mask = aal_regions == cortex_region
            modulation_values = np.nan(len(e_field_base))
            # Compute fields along the pref. direction 
            e_field_base_pref = np.multiply(e_field_base[cortex_mask], pref_dir[cortex_mask]).sum(axis=-1)
            e_field_df_pref = np.multiply(e_field_df[cortex_mask], pref_dir[cortex_mask]).sum(axis=-1)
            # Compute TI field along pref. direction inisde the cortex
            modulation_values[cortex_mask] = 2 * np.minimum(np.abs(e_field_base_pref), np.abs(e_field_df_pref))
            # Compute TI field along free direction outside the cortex
            modulation_values[np.logical_not(cortex_mask)] = modulation_envelope(e_field_base[np.logical_not(cortex_mask)], e_field_df[np.logical_not(cortex_mask)])


        # Compute max modulated field in ROI
        max_modulation_roi = np.amax(modulation_values[roi_mask])
        # Initalize sums
        roi_region_sum = 0
        non_roi_region_sum = 0
        # Iterate over regions 
        if parallel:
            def processRegion(region):
                region_mask = all_region_masks[region]
                region_sum = modulation_values[region_mask].sum() / region_volumes[region]
                return region_sum

            with ThreadPoolExecutor() as executor:
                region_sums = list(executor.map(processRegion, range(len(all_region_masks))))

            for region, region_sum in enumerate(region_sums):
                if region in regions_of_interest_ids:
                    roi_region_sum += region_sum
                else:
                    non_roi_region_sum += region_sum
        else:
            for region in range(len(all_region_masks)):
                region_mask = all_region_masks[region]
                if region in regions_of_interest_ids:
               
                    roi_region_sum += modulation_values[region_mask].sum() / region_volumes[region]
                else:
                    non_roi_region_sum += modulation_values[region_mask].sum()  / region_volumes[region]
                
    
        region_ratio = np.nan_to_num(roi_region_sum/non_roi_region_sum)
        fitness_measure = region_ratio*10000
       
        return max_modulation_roi, fitness_measure
    if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
        return 100*(np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000
    
    

    penalty = 0
    # The first 4 indices are the electrode IDs
    electrodes = np.round(x[:4]).astype(np.int32) 
    # Compute mask for ROI
    roi_mask = np.isin(aal_regions, np.array(regions_of_interest))
    # Compute region of ineterest id
    regions_of_interest_ids = [i for i, lbl in enumerate(np.unique(aal_regions)) if lbl in regions_of_interest]
    # Get binary masks for all regions
    all_region_masks = np.array([aal_regions == lbl for lbl in np.unique(aal_regions)])
    if parallel:
        with ThreadPoolExecutor() as executor:
            fitness_valsANDmax_vals = np.array(list(executor.map(processCurrent, currents)))
        max_vals = fitness_valsANDmax_vals[:,0]
        fitness_vals = fitness_valsANDmax_vals[:,1]
    else:
        fitness_vals = []
        max_vals = []
        for current in currents:
            max_val, fitness_val = processCurrent(current)
            fitness_vals.append(fitness_val)
            max_vals.append(max_val)

        max_vals = np.array(max_vals)
        fitness_vals = np.array(fitness_vals)


    return_fitness = 0
    if not np.any(max_vals > threshold):
        penalty += 100*((threshold - np.mean(max_vals))**2) + 1000
        return_fitness = np.amin(fitness_vals)
    else:
        # NOTE: Check
        fitness_candidate = np.amax(fitness_vals[max_vals > threshold])
        return_fitness = fitness_candidate
    
    return -float(np.round(return_fitness - penalty, 2))

def objective_df_np(x, field_data, regions_of_interest, aal_regions, region_volumes, currents, threshold, mode='free'):  
    def processCurrents(currents):
        assert len(currents.shape) == 2 and currents.shape[1] == 2, 'currents must have the shape (# of current choices, 2)'
        # Number of coices
        n_currents = currents.shape[0]
        # Number of elements
        n_elems = field_data.shape[1]
        # Compute field for freq 1
        e_field_base = np.expand_dims(currents[:,0:1], axis=-1)*field_data[electrodes[0]] - np.expand_dims(currents[:,0:1], axis=-1)*field_data[electrodes[1]]
        # Reshape field
        e_field_base = np.resize(e_field_base, [n_currents * n_elems, 3])
        # Compute field for freq 2
        e_field_df = np.expand_dims(currents[:,1:2], axis=-1)*field_data[electrodes[2]] - np.expand_dims(currents[:,1:2], axis=-1)*field_data[electrodes[3]]
        # Reshape field
        e_field_df = np.resize(e_field_df, [n_currents * n_elems, 3])
        # Compute modulated field
        if mode == 'free':
            modulation_values_flat = modulation_envelope(e_field_base, e_field_df)
        else:
            modulation_values_flat = 2 * np.minimum(np.abs(e_field_base), np.abs(e_field_df))
        # Resize 
        modulation_values = np.resize(modulation_values_flat, [n_currents, n_elems])
        
        # Compute max modulated field in ROI
        max_modulation_values_roi = np.amax(modulation_values[:, roi_mask], axis=1)
        # Apply masks(# of regions, # of currents, # of elems)
        all_modulation_values_regions = np.multiply(np.expand_dims(modulation_values, axis=0), 
                                                    np.expand_dims(all_region_masks, axis=1))
        # Number of regions
        n_regions = all_region_masks.shape[0]
        # ROI id mask
        ROI_id_mask = np.isin(np.arange(n_regions), regions_of_interest_ids)
       
        # Get modulation in ROI
        roi_region_modulation = all_modulation_values_regions[ROI_id_mask] 
        # Compute ROI sum
        roi_region_sum = roi_region_modulation.sum(axis=-1) / np.expand_dims(region_volumes[ROI_id_mask], axis=-1)
        roi_region_sums = roi_region_sum.sum(axis=0)
        # Get modulation outside ROI
        nonroi_region_modulation = all_modulation_values_regions[np.logical_not(ROI_id_mask)] 
        # Compute non-ROI sum
        non_roi_region_sum = nonroi_region_modulation.sum(axis=-1) / np.expand_dims(region_volumes[np.logical_not(ROI_id_mask)], axis=-1)
        non_roi_region_sums = non_roi_region_sum.sum(axis=0)
        # Compute ratio
        region_ratio = np.nan_to_num(roi_region_sums / non_roi_region_sums)
        fitness_measures = region_ratio*10000

        return max_modulation_values_roi, fitness_measures
    assert mode in ['free', 'pref'] and ( (mode == 'free' and len(field_data.shape) == 3 and field_data.shape[-1] == 3) or (mode == 'pref' and len(field_data.shape) == 2)), 'mode is either set incorrectly or does not match the shape nof field data'
    if np.unique(np.round(x[:4]), return_counts=True)[1].size != 4:
        return 100*(np.abs((np.unique(np.round(x[:4]), return_counts=True)[1].size - 4))**2) + 10000
    
    penalty = 0
    # The first 4 indices are the electrode IDs
    electrodes = np.round(x[:4]).astype(np.int32) 
    # Compute mask for ROI
    roi_mask = np.isin(aal_regions, np.array(regions_of_interest))
    # Get binary masks for all regions
    all_region_masks = np.array([aal_regions == lbl for lbl in np.unique(aal_regions)])
    # Compute region of ineterest id
    regions_of_interest_ids = [i for i, lbl in enumerate(np.unique(aal_regions)) if lbl in regions_of_interest]
    # Compute metrics
    max_vals, fitness_vals = processCurrents(currents)
  
    max_vals = np.array(max_vals)
    fitness_vals = np.array(fitness_vals)
    return_fitness = 0
    if not np.any(max_vals > threshold):
        penalty += 100*((threshold - np.mean(max_vals))**2) + 1000
        return_fitness = np.amin(fitness_vals)
    else:
        # NOTE: Check
        fitness_candidate = np.amax(fitness_vals[max_vals > threshold])
        return_fitness = fitness_candidate
    
    return -float(np.round(return_fitness - penalty, 2))

def runGA(field_data, roi_labels, aal_regions, region_volumes, algorithm_param, cur_min=5e-1, cur_max=15e-1, cur_step=.25, Imax=2, opt_threshold=2e-1, pref_dir='None', cortex_region=None, mode='free', parallel=False, gpu=False):
    # # Set currents
    # cur_potential_values = np.arange(cur_min, cur_max, cur_step)

    # # Create meshgrid
    # cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
    # # Create combos 
    # cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))

    # Usable currents 
    currents1 = np.expand_dims(np.arange(cur_min, cur_max, cur_step), axis=-1)
    currents2 = Imax * np.ones_like(currents1)
    usable_currents = np.concatenate([currents1, currents2], 
                                axis=-1)
    usable_currents = usable_currents[usable_currents[:,0] <= usable_currents[:,1],:]
 
    # Debugging: prints currents 
    print(usable_currents)
    # Objective function 
    if gpu:
        aal_regions = cp.array(aal_regions)
        ga_objective_df = lambda x, **kwargs: objective_df_cp(x, 
                                                        field_data, 
                                                        roi_labels, 
                                                        aal_regions, 
                                                        region_volumes, 
                                                        usable_currents, 
                                                        opt_threshold,
                                                        mode=mode)
    else:
        ga_objective_df = lambda x, **kwargs: objective_df(x, 
                                                        field_data, 
                                                        roi_labels, 
                                                        aal_regions, 
                                                        region_volumes, 
                                                        usable_currents, 
                                                        opt_threshold,
                                                        pref_dir=pref_dir,
                                                        cortex_region=cortex_region,
                                                        mode=mode,
                                                        parallel=parallel
                                                        )
        
    # Set variable type
    var_type = np.array([['int']]*4)
    # Number of electrodes
    n_elecs = field_data.shape[0]
    # Lower and upper bounds 
    bounds = np.array([[0, n_elecs-1]]*4)
    # Run GA 
    result = ga(function=ga_objective_df, 
                dimension=bounds.shape[0], 
                variable_type_mixed=var_type, 
                variable_boundaries=bounds, 
                algorithm_parameters=algorithm_param, 
                function_timeout=120., 
                convergence_curve=False)
    result.run()
    print('GA finished')
    # Convergence 
    convergence = result.report
    # Solution 
    solution = result.output_dict
    # Optimal electrodes
    opt_elecs = solution['variable'].astype(np.int32)
    # Initialize 
    best_f = float('inf')
    best_curr = usable_currents[0]
    # Loop over currents 
    for i in range(len(usable_currents)):
        curr_f = objective_df(opt_elecs, 
                    field_data, 
                    roi_labels, 
                    aal_regions, 
                    region_volumes, 
                    usable_currents[i:i+1], 
                    opt_threshold,
                    mode=mode,
                    parallel=parallel)
       
        
        if curr_f < best_f:
            best_curr = usable_currents[i]
            best_f = curr_f
    
    # Solution  (add 1 as Python is 0-indexed)
    sol = {'electrodes': opt_elecs + 1, 'currents': best_curr}
    
    return sol, convergence
   

if __name__ == "__main__":
    OPTIMIZATION_THRESHOLD = 0.2
    # Define datapath    
    # data_path = '/Users/arminmoharrer/Library/CloudStorage/GoogleDrive-armin.moharrer@umb.edu/My Drive/TIS_Code/Data'
    # npz_arrays = np.load(os.path.join(data_path, 'sphere_19/data.npz'), 
    #                      allow_pickle=True)

    # field_data = npz_arrays['e_field']
    # prefVec = scipy.io.loadmat(os.path.join(data_path, 'Sphere_19/prefdir.mat'))['prefDir']
    # n_elecs = field_data.shape[0]
    # print(f"There are {n_elecs} electrodes.")
    # # Set GA parameters 
    # algorithm_param = {'max_num_iteration': 25,
    #                    'population_size': 100,
    #                    'mutation_probability': 0.4,
    #                    'elit_ratio': 0.05,
    #                    'crossover_probability': 0.5,
    #                    'parents_portion': 0.2,
    #                    'crossover_type': 'uniform',
    #                    'max_iteration_without_improv': None
    #                 }
    
    
    # # Load ROI
    # for roi_file in os.listdir(os.path.join(data_path, 'sphere_19/ROI')):
    #     if '0_0_0' not in roi_file:
    #         continue 
    #     # Load ROI mask
    #     aal_regions = np.squeeze(scipy.io.loadmat(os.path.join(data_path, f'sphere_19/ROI/{roi_file}'))['roi_mask']) 
    #     # Compute region volumes
    #     region_volumes = np.array([sum(aal_regions == lbl) for lbl in np.unique(aal_regions)])
    #     # Set ROI label 
    #     roi_labels = np.array([1])
    #     # File prefix
    #     prefix = f'sol_GA{'_'.join([key + '_' + str(algorithm_param[key]) for key in algorithm_param])}_opthrsh_{OPTIMIZATION_THRESHOLD}_{roi_file[:-4]}'
    #     # Set currents
    #     cur_potential_values = np.arange(.5, 4, .5)

    #     # Create meshgrid
    #     cur_x, cur_y = np.meshgrid(cur_potential_values, cur_potential_values)
    #     # Create combos 
    #     cur_all_combinations = np.hstack((cur_x.reshape((-1, 1)), cur_y.reshape((-1, 1))))
    #     # Usable currents 
    #     usable_currents = cur_all_combinations[np.where(np.sum(np.round(cur_all_combinations, 2), axis=1) == 4)[0]]

    #     ga_objective_df = lambda x, **kwargs: objective_df(x, 
    #                                                 field_data, 
    #                                                 roi_labels, 
    #                                                 aal_regions, 
    #                                                 region_volumes, 
    #                                                 usable_currents, 
    #                                                 .2,
    #                                                 parallel=False)
    #     ga_objective_np = lambda x, **kwargs: objective_df_np(x, 
    #                                                 field_data, 
    #                                                 roi_labels, 
    #                                                 aal_regions, 
    #                                                 region_volumes, 
    #                                                 usable_currents, 
    #                                                 .2,
    #                                                )
        
    #     y = ga_objective_np([1,2,3,4])
        
    #     print(f'Done for {roi_file}')
    # print("Happy Optimizing!")
