import sys,os,os.path
sys.path.append("/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/Olivia/")
sys.path.append("/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/Users/ianosborne/Desktop/University_of_Manchester_Physics.nosync/Masters/Olivia/'

import json
import numpy as np

from enum import Enum

from olivia          .hist_io        import save_histomanager_to_file
from invisible_cities.core.configure import configure
import olivia.monitor_functions          as monf


class InputDataType(Enum):
    rwf   = 0
    pmaps = 1


def olivia(conf):
    files_in     = os.path.expandvars(conf.files_in)
    file_out     = os.path.expandvars(conf.file_out)
    detector_db  =                    conf.detector_db
    run_number   =                int(conf.run_number)
    histo_config = os.path.expandvars(conf.histo_config)

    try:
        data_type = InputDataType[conf.data_type]
    except KeyError:
        print(f'Error: Data type {conf.data_type} is not recognized.')
        raise

    with open(histo_config) as config_file:
        config_dict = json.load(config_file)
        if   data_type == InputDataType.rwf  :
            histo_manager = monf.fill_rwf_histos (files_in, config_dict)
            pmt_monitoring = monf.pmt_monitoring_test(files_in)

        elif data_type == InputDataType.pmaps:
            histo_manager = monf.fill_pmap_histos(files_in, detector_db,
                                                  run_number, config_dict)

        save_histomanager_to_file(histo_manager, file_out)

    


if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    olivia(conf)
