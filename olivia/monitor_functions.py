import sys,os,os.path
sys.path.append("/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/Olivia/")
sys.path.append("/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/Olivia/'



import glob
from   collections import defaultdict

import numpy  as np
import tables as tb

from olivia        import histogram_functions as histf
from olivia.histos import        HistoManager

from invisible_cities.database import         load_db as   dbf
from invisible_cities.core     import system_of_units as units

from invisible_cities.io  .pmaps_io                import      load_pmaps
from invisible_cities.io  .dst_io                  import        load_dst
from invisible_cities.core.tbl_functions           import get_vectors
from invisible_cities.calib.calib_sensors_functions import           modes
from invisible_cities.calib.calib_functions         import      SensorType



def pmap_bins(config_dict):
    """Generates the binning arrays, label and scale of the monitor plots
    from the config dictionary that contains the ranges, number of bins,
    labels and scales.

    Returns a dictionary with the bins, another with the labels and
    another with the scales.
    """
    var_bins   = {}
    var_labels = {}
    var_scales = {}

    for k, v in config_dict.items():
        if   "_bins"   in k: var_bins  [k.replace("_bins"  , "")] = [np.linspace(v[0], v[1], v[2] + 1)]
        elif "_labels" in k: var_labels[k.replace("_labels", "")] = v
        elif "_scales" in k: var_scales[k.replace("_scales", "")] = v

    exception = ['S1_Energy', 'S1_Number', 'S1_Time']
    bin_sel   = lambda x: ('S2' not in x) and (x not in exception)
    for param in filter(bin_sel, list(var_bins)):
        var_bins  ['S1_Energy_' + param] = var_bins  ['S1_Energy'] + \
                                           var_bins  [param]
        var_labels['S1_Energy_' + param] = var_labels['S1_Energy'] + \
                                           var_labels[param]
        var_scales['S1_Energy_' + param] = var_scales['S1_Energy']
    var_bins      ['S1_Time_S1_Energy']  = var_bins  ['S1_Time'] + \
                                           var_bins  ['S1_Energy']
    var_labels    ['S1_Time_S1_Energy']  = var_labels['S1_Time'] + \
                                           var_labels['S1_Energy']
    var_scales    ['S1_Time_S1_Energy']  = var_scales['S1_Time']

    exception = ['S2_Energy', 'S2_Number', 'S2_Time']
    bin_sel   = lambda x: ('S1' not in x) and (x not in exception) and ('SiPM' not in x)
    for param in filter(bin_sel, list(var_bins)):
        var_bins  ['S2_Energy_' + param]  = var_bins  ['S2_Energy'] + \
                                            var_bins  [param]
        var_labels['S2_Energy_' + param]  = var_labels['S2_Energy'] + \
                                            var_labels[param]
        var_scales['S2_Energy_' + param]  = var_scales['S2_Energy']

    var_bins      ['S2_Time_S2_Energy']   = var_bins  ['S2_Time']   + \
                                            var_bins  ['S2_Energy']
    var_labels    ['S2_Time_S2_Energy']   = var_labels['S2_Time']   + \
                                            var_labels['S2_Energy']
    var_scales    ['S2_Time_S2_Energy']   = var_scales['S2_Time']

    var_bins      ['S2_Energy_S1_Energy'] = var_bins  ['S2_Energy'] + \
                                            var_bins  ['S1_Energy']
    var_labels    ['S2_Energy_S1_Energy'] = var_labels['S2_Energy'] + \
                                            var_labels['S1_Energy']
    var_scales    ['S2_Energy_S1_Energy'] = var_scales['S2_Energy']

    var_bins      ['S2_XYSiPM']           = var_bins  ['S2_XSiPM']  + \
                                            var_bins  ['S2_YSiPM']
    var_labels    ['S2_XYSiPM']           = var_labels['S2_XSiPM']  + \
                                            var_labels['S2_YSiPM']
    var_scales    ['S2_XYSiPM']           = var_scales['S2_XSiPM']

    for i in range(config_dict['nPMT']):
        var_bins  [f'PMT{i}_S2_Energy'] =               var_bins  ['S2_Energy']
        var_bins  [f'PMT{i}_S2_Height'] =               var_bins  ['S2_Height']
        var_bins  [f'PMT{i}_S2_Time'  ] =               var_bins  ['S2_Time'  ]
        var_labels[f'PMT{i}_S2_Energy'] = [f'PMT{i} ' + var_labels['S2_Energy'][0]]
        var_labels[f'PMT{i}_S2_Height'] = [f'PMT{i} ' + var_labels['S2_Height'][0]]
        var_labels[f'PMT{i}_S2_Time'  ] = [f'PMT{i} ' + var_labels['S2_Time'  ][0]]
        var_scales[f'PMT{i}_S2_Energy'] =               var_scales['S2_Energy']
        var_scales[f'PMT{i}_S2_Height'] =               var_scales['S2_Height']
        var_scales[f'PMT{i}_S2_Time'  ] =               var_scales['S2_Time']

    del var_bins['S2_XSiPM']
    del var_bins['S2_YSiPM']

    return var_bins, var_labels, var_scales


def fill_pmap_var_1d(speaks, var_dict, ptype, DataSiPM=None):
    """Fills a passed dictionary of lists with the pmap variables to monitor.

    Parameters
    ----------
    speaks   : sequence
    List of S1 or S2s.
    var_dict : dict
    Stores the variable values.
    ptype    : str
    Type of pmap ('S1' or 'S2')
    DataSiPM : Database
    Contains the SiPM information. Only needed in case of 'S2' ptype
    """
    var_dict[ptype + '_Number'].append(len(speaks))
    for speak in speaks:
        var_dict[ptype + '_Width' ].append(speak.width / units.mus)
        var_dict[ptype + '_Height'].append(speak.height)
        var_dict[ptype + '_Energy'].append(speak.total_energy)
        var_dict[ptype + '_Charge'].append(speak.total_charge)
        var_dict[ptype + '_Time'  ].append(speak.time_at_max_energy / units.mus)

        if ptype == 'S2':
            nS1 = var_dict['S1_Number'][-1]
            var_dict    [ptype + '_SingleS1']       .append(nS1)
            if nS1 == 1:
                var_dict[ptype + '_SingleS1_Energy'].append(var_dict['S1_Energy'][-1])

            sipm_ids = speak.sipms.ids
            sipm_Q   = speak.sipms.sum_over_times
            
            var_dict    [ptype + '_NSiPM' ].append(len(sipm_ids))
            var_dict    [ptype + '_QSiPM' ].extend(sipm_Q)
            var_dict    [ptype + '_IdSiPM'].extend(sipm_ids)
            if len(sipm_ids) > 0:
                var_dict[ptype + '_XSiPM' ].extend(DataSiPM.X.values[sipm_ids])
                var_dict[ptype + '_YSiPM' ].extend(DataSiPM.Y.values[sipm_ids])


def fill_pmap_var_2d(var_dict, ptype):
    """Makes 2d combinations of the variables stored in a dictionary
    that contains the pmaps variables.

    Parameters
    ----------
    var_dict : dict
    Stores the variable values.
    ptype    : str
    Type of pmap ('S1' or 'S2')
    """
    param_list = ['Width', 'Height', 'Charge']
    for param in param_list:
        var_dict[ptype + '_Energy_' + ptype + '_' + param] = np.array([var_dict[ptype + '_Energy'], var_dict[ptype + '_' + param]])
    var_dict    [ptype + '_Time_' + ptype + '_Energy']     = np.array([var_dict[ptype + '_Time']  , var_dict[ptype + '_Energy']])

    if ptype == 'S2':
        sel = np.asarray(var_dict['S2_SingleS1']) == 1
        var_dict[ptype + '_Energy_S1_Energy'] = np.array([np.asarray(var_dict[ptype + '_Energy'])[sel],
                                                          np.asarray(var_dict[ptype + '_SingleS1_Energy'])])
        var_dict[ptype + '_XYSiPM']           = np.array([var_dict[ptype + '_XSiPM'], var_dict[ptype + '_YSiPM']])


def fill_pmt_var(speaks, var_dict):
    for speak in speaks:
        pmts     = speak.pmts
        times    = speak.times
        energies = pmts .sum_over_times
        heights  = np.max             (pmts.all_waveforms, axis=1)
        times    = np.apply_along_axis(lambda wf: times[np.argmax(wf)], axis=1, arr=pmts.all_waveforms) / units.mus

        for i in range(len(pmts.all_waveforms)):
            var_dict[f'PMT{i}_S2_Energy'].append(energies[i])
            var_dict[f'PMT{i}_S2_Height'].append(heights [i])
            var_dict[f'PMT{i}_S2_Time'  ].append(times   [i])


def fill_pmap_var(pmap, sipm_db):
    var = defaultdict(list)

    fill_pmap_var_1d(pmap.s1s, var, 'S1')
    fill_pmap_var_1d(pmap.s2s, var, 'S2', sipm_db)
    fill_pmap_var_2d(var, 'S1')
    fill_pmap_var_2d(var, 'S2')
    fill_pmt_var    (pmap.s2s, var)

    del var['S2_XSiPM']
    del var['S2_YSiPM']
    del var['S2_SingleS1']
    del var['S2_SingleS1_Energy']

    return var


def fill_pmap_histos(in_path, detector_db, run_number, config_dict):
    """Creates and returns a HistoManager object with the pmap histograms.

    Parameters
    ----------
    in_path     : str
    Path to the file(s) to be monitored.
    run_number  : int
    Run number of the dataset (used to obtain the SiPM database).
    config_dict : dict
    Contains the configuration parameters (bins, labels).
    """
    var_bins, var_labels, var_scales = pmap_bins(config_dict)
    histo_manager        = histf.create_histomanager_from_dicts(var_bins  ,
                                                                var_labels,
                                                                var_scales)
    SiPM_db              = dbf.DataSiPM(detector_db, run_number)

    for in_file in glob.glob(in_path):
        pmaps = load_pmaps(in_file)
        for pmap in pmaps.values():
            var = fill_pmap_var(pmap, SiPM_db)
            histo_manager.fill_histograms(var)
    return histo_manager


def rwf_bins(config_dict):
    """Generates the binning arrays, label and scale of the rwf monitor
    plots from the a config dictionary that contains the ranges,
    number of bins, labels and scales.

    Returns a dictionary with the bins, another with the labels and
    another with the scales.
    """
    var_bins   = {}
    var_labels = {}
    var_scales = {}

    for k, v in config_dict.items():
        if "Raw_ADC_counts" in k: continue

        if   "_bins"   in k: var_bins  [k.replace("_bins"  , "")] = [np.linspace(v[0], v[1], v[2] + 1)]
        elif "_labels" in k: var_labels[k.replace("_labels", "")] = v
        elif "_scales" in k: var_scales[k.replace("_scales", "")] = v

    #n_PMTs = config_dict["n_PMTs"]
    #v      = config_dict["Raw_ADC_counts_bins"]
    #for i in range(0, int(n_PMTs)):
    #    var_bins  [f"PMT{i}_ADCs"] = [np.linspace(v[0], v[1], v[2] + 1)]
    #    var_labels[f"PMT{i}_ADCs"] = [f"PMT{i}_Raw_ADC_counts"]
    #    var_scales[f"PMT{i}_ADCs"] = config_dict["Raw_ADC_counts_scales"]

    return var_bins, var_labels, var_scales, config_dict['n_baseline']


def fill_rwf_var(rwfs, var_dict, sensor_type):
    """Fills a passed dictionary of lists with the rwf variables to monitor.

    Parameters
    ----------
    rwf         : dataframe
    Raw waveforms.
    var_dict    : dict
    Stores the variable values.
    sensor_type : AutoNameEnumBase
    Type of sensor: SensorType.SIPM or SensorType.PMT.
    """

    if   sensor_type is SensorType.SIPM:
        bls = modes(rwfs.astype("int16")).flatten()
        rms = [np.std(rwf[rwf>0][:int(0.4*len(rwf))]) for rwf in rwfs if len(rwf[rwf>0])]
    elif sensor_type is SensorType.PMT:
        bls = np.mean(rwfs, axis=1)
        rms = np.std(rwfs[:, :int(0.4*len(rwfs[0]))], axis=1)

    var_dict[sensor_type.name + '_Baseline']   .extend(bls)
    var_dict[sensor_type.name + '_BaselineRMS'].extend(rms)
    var_dict[sensor_type.name + '_nSensors']   .append(len(bls))

    #PMT_#_ACD plots
    #if sensor_type is SensorType.PMT:
    #    for i in range(0, len(rwfs)):
    #        var_dict[f'PMT{i}_ADCs'].extend(rwfs[i])


def fill_rwf_histos(in_path, config_dict):
    """Creates and returns a HistoManager object with the waveform histograms.

    Parameters
    ----------
    in_path     : str
    Path to the file(s) to be monitored.
    config_dict : dict
    Contains the configuration parameters (bins, labels).
    """
    var_bins, var_labels, var_scales, n_baseline = rwf_bins(config_dict)

    histo_manager = histf.create_histomanager_from_dicts(var_bins  ,
                                                         var_labels,
                                                         var_scales)

    for in_file in glob.glob(in_path):
        with tb.open_file(in_file, "r") as h5in:
            var = defaultdict(list)
            pmtrwf, pmtblr, sipmrwf = get_vectors(h5in)
            nevt = len(pmtrwf)
            for evt in range(nevt):
                fill_rwf_var(pmtrwf [evt, :, :n_baseline], var, SensorType. PMT)
                fill_rwf_var(sipmrwf[evt]                , var, SensorType.SIPM)
                
        histo_manager.fill_histograms(var)
    return histo_manager


# def kdst_bins(config_dict):
#     """Generates the binning arrays and label of the kdst monitor plots from the
#     config dictionary that contains the ranges, number of bins and labels.
#
#     Returns a dictionary with the bins and another with the labels.
#     """
#     var_bins   = {}
#     var_labels = {}
#
#     for k, v in config_dict.items():
#         if   "_bins"   in k: var_bins  [k.replace("_bins"  , "")] = [np.linspace(v[0], v[1], v[2] + 1)]
#         elif "_labels" in k: var_labels[k.replace("_labels", "")] = v
#
#     var_bins  ['S2e_Z'  ] = var_bins  ['Z'  ] + var_bins  ['S2e']
#     var_labels['S2e_Z'  ] = var_labels['Z'  ] + var_labels['S2e']
#
#     var_bins  ['S2q_Z'  ] = var_bins  ['Z'  ] + var_bins  ['S2q']
#     var_labels['S2q_Z'  ] = var_labels['Z'  ] + var_labels['S2q']
#
#     var_bins  ['S2e_R'  ] = var_bins  ['R'  ] + var_bins  ['S2e']
#     var_labels['S2e_R'  ] = var_labels['R'  ] + var_labels['S2e']
#
#     var_bins  ['S2q_R'  ] = var_bins  ['R'  ] + var_bins  ['S2q']
#     var_labels['S2q_R'  ] = var_labels['R'  ] + var_labels['S2q']
#
#     var_bins  ['S2e_Phi'] = var_bins  ['Phi'] + var_bins  ['S2e']
#     var_labels['S2e_Phi'] = var_labels['Phi'] + var_labels['S2e']
#
#     var_bins  ['S2q_Phi'] = var_bins  ['Phi'] + var_bins  ['S2q']
#     var_labels['S2q_Phi'] = var_labels['Phi'] + var_labels['S2q']
#
#     var_bins  ['S2e_X'  ] = var_bins  ['X'  ] + var_bins  ['S2e']
#     var_labels['S2e_X'  ] = var_labels['X'  ] + var_labels['S2e']
#
#     var_bins  ['S2q_X'  ] = var_bins  ['X'  ] + var_bins  ['S2q']
#     var_labels['S2q_X'  ] = var_labels['X'  ] + var_labels['S2q']
#
#     var_bins  ['S2e_Y'  ] = var_bins  ['Y'  ] + var_bins  ['S2e']
#     var_labels['S2e_Y'  ] = var_labels['Y'  ] + var_labels['S2e']
#
#     var_bins  ['S2q_Y'  ] = var_bins  ['Y'  ] + var_bins  ['S2q']
#     var_labels['S2q_Y'  ] = var_labels['Y'  ] + var_labels['S2q']
#
#     var_bins  ['XY'     ] = var_bins  ['X'  ] + var_bins  ['Y'  ]
#     var_labels['XY'     ] = var_labels['X'  ] + var_labels['Y'  ]
#
#     var_bins  ['S2e_XY'] = var_bins  ['X'  ] + var_bins  ['Y'  ] + var_bins  ['S2e']
#     var_labels['S2e_XY'] = var_labels['X'  ] + var_labels['Y'  ] + var_labels['S2e']
#
#     var_bins  ['S2q_XY'] = var_bins  ['X'  ] + var_bins  ['Y'  ] + var_bins  ['S2q']
#     var_labels['S2q_XY'] = var_labels['X'  ] + var_labels['Y'  ] + var_labels['S2q']
#
#     return var_bins, var_labels
#
#
# def fill_kdst_var_1d(kdst, var_dict):
#     """Fills a passed dictionary of lists with the kdst variables to monitor.
#
#     Parameters
#     ----------
#     kdst     : kdst dataframe.
#     var_dict : dict
#     Stores the variable values.
#     """
#
#     var_names = kdst.keys()
#
#     var_sel       = lambda x: (x not in ['event', 'time', 'peak'])
#     var_ns_to_mus = ['S1t', 'S2t', 'S1w']
#
#     for param in filter(var_sel, list(var_names)):
#         values = kdst[param].values
#         if param in var_ns_to_mus:
#             values = values / units.mus
#         var_dict[param].extend(values)
#
#
# def fill_kdst_var_2d(var_dict):
#     """
#     Makes 2d combinations of the variables stored in a dictionary that contains
#     the kdst variables.
#
#     Arguments:
#     var_dict = Dictionary that stores the variable values.
#     """
#     param_list = ['Z', 'X', 'Y', 'R', 'Phi']
#     for param in param_list:
#         var_dict['S2e_' + param] = np.array([var_dict[param], var_dict['S2e']])
#         var_dict['S2q_' + param] = np.array([var_dict[param], var_dict['S2q']])
#     var_dict    ['XY'          ] = np.array([var_dict['X'  ], var_dict['Y'  ]])
#     var_dict    ['S2e_XY'      ] = np.array([var_dict['X'  ], var_dict['Y'  ], var_dict['S2e']])
#     var_dict    ['S2q_XY'      ] = np.array([var_dict['X'  ], var_dict['Y'  ], var_dict['S2q']])
#
# def fill_kdst_histos(in_path, config_dict):
#     """
#     Creates and returns an HistoManager object with the kdst histograms.
#
#     Arguments:
#     in_path     = String with the path to the file(s) to be monitored.
#     config_dict = Dictionary with the configuration parameters (bins, labels)
#     """
#     var_bins, var_labels = kdst_bins(config_dict)
#
#     histo_manager = histf.create_histomanager_from_dicts(var_bins, var_labels)
#
#     for in_file in glob.glob(in_path):
#         var  = defaultdict(list)
#         kdst = load_dst   (in_file, 'DST', 'Events')
#
#         fill_kdst_var_1d  (kdst, var)
#         fill_kdst_var_2d  (var)
#
#         histo_manager.fill_histograms(var)
#     return histo_manager
